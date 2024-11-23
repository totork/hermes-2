/*

    Copyright B.Dudson, J.Leddy, University of York, 2016-2019
              email: benjamin.dudson@york.ac.uk

    This file is part of Hermes-2 (Hot ion version)

    Hermes is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Hermes is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Hermes.  If not, see <http://www.gnu.org/licenses/>.

*/
#include "hermes-2.hxx"

#include <bout/derivs.hxx>
#include <bout/field_factory.hxx>
#include <bout/initialprofiles.hxx>

#include <bout/invert_parderiv.hxx>
#include <bout/parallel_boundary_region.hxx>
#include <bout/boundary_region.hxx>

#include "div_ops.hxx"
#include "loadmetric.hxx"

#include <bout/constants.hxx>
#include <bout/assert.hxx>
#include <bout/fv_ops.hxx>
#include <cmath>


// OpenADAS interface Atomicpp by T.Body
#include "atomicpp/ImpuritySpecies.hxx"
#include "atomicpp/Prad.hxx"

std::string parbc{"parallel_neumann_o1"};


template <typename T>
T max_abs(T a, T b) {
    return (std::abs(a) > std::abs(b)) ? a : b;
}

// Recursive case for more than two arguments                                                                                                                                                                      
template <typename T, typename... Args>
T max_abs(T first, Args... args) {
    return max_abs(first, max_abs(args...));
}


BoutReal floor(BoutReal var, BoutReal f) {
  if (var < f)
    return f;
  return var;
}

/// Returns a copy of input \p var with all values greater than \p f replaced by
/// \p f.
const Field3D ceil(const Field3D &var, BoutReal f, REGION rgn = RGN_ALL) {
  checkData(var);
  Field3D result = copy(var);

  BOUT_FOR(d, var.getRegion(rgn)) {
    if (result[d] > f) {
      result[d] = f;
    }
  }

  return result;
}

bool isZero(const Field3D &f) {
  const auto lmin = min(f, true);
  const auto lmax = max(f, true);
  return (lmin == 0.0 && lmax == 0.0);
}

// Square function for vectors
Field3D SQ(const Vector3D &v) { return v * v; }

void setRegions(Field3D &f) {
  f.yup().setRegion("RGN_YPAR_+1");
  f.ydown().setRegion("RGN_YPAR_-1");
}

const Field3D &yup(const Field3D &f) { return f.yup(); }
BoutReal yup(BoutReal f) { return f; };
const Field3D &ydown(const Field3D &f) { return f.ydown(); }
BoutReal ydown(BoutReal f) { return f; };
const BoutReal yup(BoutReal f, Ind3D i) { return f; };
const BoutReal ydown(BoutReal f, Ind3D i) { return f; };
// const BoutReal& yup(const Field3D &f, Ind3D i) { return f.yup()[i.yp()]; }
// const BoutReal& ydown(const Field3D &f, Ind3D i) { return f.ydown()[i.ym()];
// } BoutReal& yup(Field3D &f, Ind3D i) { return f.yup()[i.yp()]; } BoutReal&
// ydown(Field3D &f, Ind3D i) { return f.ydown()[i.ym()]; }
const BoutReal &yup(const Field3D &f, Ind3D i) { return f.yup()[i]; }
const BoutReal &ydown(const Field3D &f, Ind3D i) { return f.ydown()[i]; }
BoutReal &yup(Field3D &f, Ind3D i) { return f.yup()[i]; }
BoutReal &ydown(Field3D &f, Ind3D i) { return f.ydown()[i]; }
const BoutReal &_get(const Field3D &f, Ind3D i) { return f[i]; }
BoutReal &_get(Field3D &f, Ind3D i) { return f[i]; }
BoutReal _get(BoutReal f, Ind3D i) { return f; };
BoutReal copy(BoutReal f) { return f; };

void alloc_all(Field3D &f) {
  f.allocate();
  f.splitParallelSlices();
  f.yup().allocate();
  f.ydown().allocate();
  setRegions(f);
}

#define GET_ALL(name)                                                          \
  auto *name##a = &name[Ind3D(0)];                                             \
  auto *name##b = &name.yup()[Ind3D(0)];                                       \
  auto *name##c = &name.ydown()[Ind3D(0)];

#define DO_ALL(op, name)                                                       \
  template <class A, class B> Field3D name##_all(const A &a, const B &b) {     \
    Field3D result;                                                            \
    alloc_all(result);                                                         \
    BOUT_FOR(i, result.getRegion("RGN_ALL")) { name##_all(result, a, b, i); }  \
    setRegions(result);                                                        \
    return result;                                                             \
  }                                                                            \
  template <class A, class B>                                                  \
  void name##_all(Field3D &result, const A &a, const B &b, Ind3D i) {          \
    result[i] = op(_get(a, i), _get(b, i));                                    \
    yup(result, i) = op(yup(a, i), yup(b, i));                                 \
    ydown(result, i) = op(ydown(a, i), ydown(b, i));                           \
  }                                                                            \
  template <class B> void name##_all(Field3D &result, const B &b, Ind3D i) {   \
    result[i] = op(result[i], _get(b, i));                                     \
    yup(result, i) = op(yup(result, i), yup(b, i));                            \
    ydown(result, i) = op(ydown(result, i), ydown(b, i));                      \
  }

DO_ALL(floor, floor)
DO_ALL(pow, pow)

#undef DO_ALL
#define DO_ALL(op, name)                                                       \
  template <class A, class B> Field3D name##_all(const A &a, const B &b) {     \
    Field3D result;                                                            \
    alloc_all(result);                                                         \
    BOUT_FOR(i, result.getRegion("RGN_ALL")) { name##_all(result, a, b, i); }  \
    checkData(result, "RGN_ALL");                                              \
    setRegions(result);                                                        \
    return result;                                                             \
  }                                                                            \
  Field3D name##_all(const Field3D &a, const Field3D &b) {                     \
    Field3D result;                                                            \
    alloc_all(result);                                                         \
    const int n = result.getNx() * result.getNy() * result.getNz();            \
    GET_ALL(result);                                                           \
    GET_ALL(a);                                                                \
    GET_ALL(b);                                                                \
    BOUT_OMP(omp parallel for simd)                                            \
    for (int i = 0; i < n; ++i) {                                              \
      resulta[i] = aa[i] op ba[i];                                             \
      resultb[i] = ab[i] op bb[i];                                             \
      resultc[i] = ac[i] op bc[i];                                             \
    }                                                                          \
    setRegions(result);                                                        \
    return result;                                                             \
  }                                                                            \
  Field3D name##_all(const Field3D &a, BoutReal b) {                           \
    Field3D result;                                                            \
    alloc_all(result);                                                         \
    const int n = result.getNx() * result.getNy() * result.getNz();            \
    GET_ALL(result);                                                           \
    GET_ALL(a);                                                                \
    BOUT_OMP(omp parallel for simd)                                            \
    for (int i = 0; i < n; ++i) {                                              \
      resulta[i] = aa[i] op b;                                                 \
      resultb[i] = ab[i] op b;                                                 \
      resultc[i] = ac[i] op b;                                                 \
    }                                                                          \
    setRegions(result);                                                        \
    return result;                                                             \
  }                                                                            \
  template <class A, class B>                                                  \
  void name##_all(Field3D &result, const A &a, const B &b, Ind3D i) {          \
    result[i] = _get(a, i) op _get(b, i);                                      \
    yup(result, i) = yup(a, i) op yup(b, i);                                   \
    ydown(result, i) = ydown(a, i) op ydown(b, i);                             \
  }

// void div_all(Field3D & result, const Field3D & a, const Field3D & b, Ind3D i)
// {
//   result[i] = a[i] / b[i];
//   result.yup()[i.yp()] = a.yup()[i.yp()] / b.yup()[i.yp()];
//   result.ydown()[i.ym()] = a.ydown()[i.ym()] / b.ydown()[i.ym()];
// }

//#include "mul_all.cxx"
DO_ALL(*, mul)
DO_ALL(/, div)
DO_ALL(+, add)
DO_ALL(-, sub)
#undef DO_ALL

#define DO_ALL(op, name)                                                       \
  template <class A, class B> Field3D &name##_all_inp(A &a, const B &b) {      \
    BOUT_FOR(i, a.getRegion("RGN_ALL")) { name##_all(a, b, i); }               \
    checkData(a, "RGN_ALL");                                                   \
    return a;                                                                  \
  }                                                                            \
  Field3D &name##_all_inp(Field3D &a, const Field3D &b) {                      \
    const int n = a.getNx() * a.getNy() * a.getNz();                           \
    GET_ALL(a);                                                                \
    GET_ALL(b);                                                                \
    BOUT_OMP(omp parallel for simd)                                            \
    for (int i = 0; i < n; ++i) {                                              \
      aa[i] op ba[i];                                                          \
      ab[i] op bb[i];                                                          \
      ac[i] op bc[i];                                                          \
    }                                                                          \
    return a;                                                                  \
  }                                                                            \
  Field3D &name##_all_inp(Field3D &a, BoutReal b) {                            \
    const int n = a.getNx() * a.getNy() * a.getNz();                           \
    GET_ALL(a);                                                                \
    BOUT_OMP(omp parallel for simd)                                            \
    for (int i = 0; i < n; ++i) {                                              \
      aa[i] op b;                                                              \
      ab[i] op b;                                                              \
      ac[i] op b;                                                              \
    }                                                                          \
    return a;                                                                  \
  }                                                                            \
  template <class A, class B> void name##_all_inp(A &a, const B &b, Ind3D i) { \
    a[i] op _get(b, i);                                                        \
    yup(a, i) op yup(b, i);                                                    \
    ydown(a, i) op ydown(b, i);                                                \
  }

// #include "mul_all.cxx"
DO_ALL(*=, mul)
DO_ALL(/=, div)
DO_ALL(+=, add)
DO_ALL(-=, sub)

#undef DO_ALL
#define DO_ALL(op)                                                             \
  inline void op##_all(Field3D &result, const Field3D &a, Ind3D i) {           \
    result[i] = op(a[i]);                                                      \
    yup(result, i) = op(yup(a, i));                                            \
    ydown(result, i) = op(ydown(a, i));                                        \
  }                                                                            \
  inline Field3D op##_all(const Field3D &a) {                                  \
    Field3D result;                                                            \
    alloc_all(result);                                                         \
    BOUT_FOR(i, result.getRegion("RGN_ALL")) { op##_all(result, a, i); }       \
    checkData(result, "RGN_ALL");                                              \
    setRegions(result);                                                        \
    return result;                                                             \
  }

DO_ALL(sqrt)
DO_ALL(SQ)
DO_ALL(copy)
DO_ALL(exp)
DO_ALL(log)

#undef DO_ALL


void set_all(Field3D &f, BoutReal val) {
  alloc_all(f);
  BOUT_FOR(i, f.getRegion("RGN_ALL")) {
    f[i] = val;
    f.yup()[i] = val;
    f.ydown()[i] = val;
  }
}
void zero_all(Field3D &f) { set_all(f, 0); }

void check_all(Field3D &f) {
  checkData(f);
  checkData(f.yup());
  checkData(f.ydown());
}

void ASSERT_CLOSE_ALL(const Field3D &a, const Field3D &b) {
  BOUT_FOR(i, a.getRegion("RGN_NOY")) {
    ASSERT0(std::abs(a[i] - b[i]) < 1e-10);
    ASSERT0(std::abs(yup(a, i) - yup(b, i)) < 1e-10);
    ASSERT0(std::abs(ydown(a, i) - ydown(b, i)) < 1e-10);
  }
}

/// Modifies and returns the first argument, taking the boundary from second argument
/// This is used because unfortunately Field3D::setBoundary returns void
Field3D withBoundary(Field3D &&f, const Field3D &bndry) {
  f.setBoundaryTo(bndry);
  return f;
}

int Hermes::init(bool restarting) {

  auto& opt = Options::root();

  // Switches in model section
  auto& optsc = opt["Hermes"];
  auto& optne = opt["Ne"];
  auto& optnvi = opt["NVi"];
  auto& optpe = opt["Pe"];
  auto& optpi = opt["Pi"];
  auto& optvort = opt["Vort"];
  auto& optvepsi = opt["VePsi"];
  auto& opttransport = opt["Transportcoefficients"];
  auto& optnumerics = opt["Numerics"];
  auto& optsheath = opt["Sheath"];
  
  OPTION(optsc, evolve_plasma, true);
  OPTION(optsc, show_timesteps, false);
  if (BoutComm::rank() != 0) {
    show_timesteps = false;
  }

  electromagnetic = optsc["electromagnetic"]
                        .doc("Include vector potential psi in Ohm's law?")
                        .withDefault<bool>(true);

  j_pol_pi = optsc["j_pol_pi"]
              .doc("Polarisation current with explicit Pi dependence")
              .withDefault<bool>(true);

  j_pol_simplified = optsc["j_pol_simplified"]
              .doc("Polarisation current without explicit Pi dependence")
              .withDefault<bool>(false);


  //////////////////////////////////////////////////////////////////////////

  // Check which variables should be evolved

  // Electron density
  evolve_ne = optsc["evolve_ne"].doc("Evolve density?").withDefault<bool>(false);
  if (evolve_ne){
    SOLVE_FOR(Ne);
    EvolvingVars.add(Ne);
    if (output_ddt) {
      SAVE_REPEAT(ddt(Ne));
    }
  }

  // Ion momentum
  evolve_nvi = optsc["evolve_nvi"].doc("Evolve ion momentum?").withDefault<bool>(false);
  if (evolve_nvi) {
    solver->add(NVi, "NVi");
    EvolvingVars.add(NVi);
    if (output_ddt) {
      SAVE_REPEAT(ddt(NVi));
    }
  } else {
    zero_all(NVi);
  }
  
  // Electron temperature
  evolve_te = optsc["evolve_te"].doc("Evolve electron temperature?").withDefault<bool>(false);
  if (evolve_te) {
    SOLVE_FOR(Pe);
    EvolvingVars.add(Pe);
    if (output_ddt) {
      SAVE_REPEAT(ddt(Pe));
    }
  } else {
    Pe = Ne;
  }

  // Ion temperature
  evolve_ti = optsc["evolve_ti"].doc("Evolve ion temperature?").withDefault<bool>(false);
  if (evolve_ti) {
    SOLVE_FOR(Pi);
    EvolvingVars.add(Pi);
    if (output_ddt) {
      SAVE_REPEAT(ddt(Pi));
    }
  } else {
    Pi = Ne;
  }

  // Electron velocity + mag. potential

  evolve_vepsi = optsc["evolve_vepsi"].doc("Evolve electron velocity?").withDefault<bool>(false);
  if (evolve_vepsi) {
    SOLVE_FOR(VePsi);
    EvolvingVars.add(VePsi);
    if (output_ddt) {
      SAVE_REPEAT(ddt(VePsi));
    }
  } else {
    zero_all(VePsi);
  }
  
  // Vorticity
  evolve_vort = optsc["evolve_vort"].doc("Evolve Vorticity?").withDefault<bool>(false);
  if (evolve_vort) {
    SOLVE_FOR(Vort);
    EvolvingVars.add(Vort);
    if (output_ddt) {
      SAVE_REPEAT(ddt(Vort));
    }
  } else {
    zero_all(Vort);
  }
  
  //////////////////////////////////////////////////////////////////////////
  
  // Get the switches for all the terms

  
  //bool Ne_ExB, Ne_mag, Ne_parflow, Ne_collision, Ne_anomalous, Ne_sources;
  Ne_ExB = optne["Ne_ExB"].doc("Use ExB advection in density").withDefault<bool>(false);
  Ne_mag = optne["Ne_mag"].doc("Use magnetic drift in density").withDefault<bool>(false);
  Ne_parflow = optne["Ne_parflow"].doc("Use parallel flow in density").withDefault<bool>(false);
  Ne_collision = optne["Ne_collision"].doc("Use collisional transport in density").withDefault<bool>(false);
  Ne_anomalous = optne["Ne_anomalous"].doc("Use anomalous cross field transport in density").withDefault<bool>(false);
  Ne_sources = optne["Ne_sources"].doc("Use sources in density").withDefault<bool>(false);
  Ne_hyper = optne["Ne_hyper"].doc("Use hyperdiffusion in density").withDefault<bool>(false);
  Ne_numdiff = optne["Ne_numdiff"].doc("Use parallel numerical diffusion in density").withDefault<bool>(false);

  
  // bool NVi_ExB, NVi_mag, NVi_parflow, NVi_parpressure, NVi_parviscos, NVi_collision, NVi_anomalous;

  NVi_ExB = optnvi["NVi_ExB"].doc("Use ExB advection in ion momentum").withDefault<bool>(false);
  NVi_mag = optnvi["NVi_mag"].doc("Use magnetic effects in ion momentum").withDefault<bool>(false);
  NVi_parflow = optnvi["NVi_parflow"].doc("Use parallel flow effects in ion momentum").withDefault<bool>(false);
  NVi_parpressure = optnvi["NVi_parpressure"].doc("Use parallel pressure gradient in ion momentum").withDefault<bool>(false);
  NVi_parviscos = optnvi["NVi_parviscos"].doc("Use parallel viscosity in ion momentum").withDefault<bool>(false);
  NVi_collision = optnvi["NVi_collision"].doc("Use collisional effects in ion momentum").withDefault<bool>(false);
  NVi_anomalous = optnvi["NVi_anomalous"].doc("Use anomalous transport in ion momentum").withDefault<bool>(false);
  NVi_hyper = optnvi["NVi_hyper"].doc("Use hyperdiffusion in ion momentum").withDefault<bool>(false);
  NVi_numdiff = optnvi["NVi_numdiff"].doc("Use parallel numerical diffusion in ion momentum").withDefault<bool>(false);

  
  // bool Pe_ExB, Pe_mag, Pe_parflow, Pe_conduction, Pe_ohmic, Pe_thermalforce, Pe_thermalcurrent; 
  // bool Pe_collision, Pe_anomalous, Pe_sources, Pe_energyexchange;

  Pe_ExB = optpe["Pe_ExB"].doc("Use ExB advection in electron energy").withDefault<bool>(false);
  Pe_mag = optpe["Pe_mag"].doc("Use magnetic effects in electron energy").withDefault<bool>(false);
  Pe_parflow = optpe["Pe_parflow"].doc("Use parallel flow effects in electron energy").withDefault<bool>(false);
  Pe_conduction = optpe["Pe_conduction"].doc("Use thermal conduction in electron energy").withDefault<bool>(false);
  Pe_ohmic = optpe["Pe_ohmic"].doc("Include Ohmic heating in electron energy").withDefault<bool>(false);
  Pe_thermalforce = optpe["Pe_thermalforce"].doc("Include thermal force effects in electron energy").withDefault<bool>(false);
  Pe_thermalcurrent = optpe["Pe_thermalcurrent"].doc("Include thermal current effects in electron energy").withDefault<bool>(false);
  Pe_collision = optpe["Pe_collision"].doc("Include collisional effects in electron energy").withDefault<bool>(false);
  Pe_anomalous = optpe["Pe_anomalous"].doc("Include anomalous effects in electron energy").withDefault<bool>(false);
  Pe_sources = optpe["Pe_sources"].doc("Include source terms in electron energy").withDefault<bool>(false);
  Pe_energyexchange = optpe["Pe_energyexchange"].doc("Include energy exchange terms in electron energy").withDefault<bool>(false);
  Pe_hyper = optpe["Pe_hyper"].doc("Use hyperdiffusion in electron pressure").withDefault<bool>(false);
  Pe_numdiff = optpe["Pe_numdiff"].doc("Use parallel numerical diffusion in electron pressure").withDefault<bool>(false);

  // bool Pi_ExB, Pi_mag, Pi_parflow, Pi_conduction, Pi_diamagenergyexchange, Pi_parviscousheat;
  // bool Pi_resistivedrift, Pi_perpviscous, Pi_sources;

  Pi_ExB = optpi["Pi_ExB"].doc("Use ExB advection in ion energy").withDefault<bool>(false);
  Pi_mag = optpi["Pi_mag"].doc("Use magnetic effects in ion energy").withDefault<bool>(false);
  Pi_parflow = optpi["Pi_parflow"].doc("Use parallel flow effects in ion energy").withDefault<bool>(false);
  Pi_conduction = optpi["Pi_conduction"].doc("Use thermal conduction in ion energy").withDefault<bool>(false);
  Pi_diamagenergyexchange = optpi["Pi_diamagenergyexchange"].doc("Include diamagnetic energy exchange in ion energy").withDefault<bool>(false);
  Pi_parviscousheat = optpi["Pi_parviscousheat"].doc("Include parallel viscous heating in ion energy").withDefault<bool>(false);
  Pi_resistivedrift = optpi["Pi_resistivedrift"].doc("Include resistive drift effects in ion energy").withDefault<bool>(false);
  Pi_perpviscous = optpi["Pi_perpviscous"].doc("Include perpendicular viscous effects in ion energy").withDefault<bool>(false);
  Pi_sources = optpi["Pi_sources"].doc("Include source terms in ion energy").withDefault<bool>(false);
  Pi_hyper = optpi["Pi_hyper"].doc("Use hyperdiffusion in ion pressure").withDefault<bool>(false);
  Pi_numdiff = optpi["Pi_numdiff"].doc("Use parallel numerical diffusion in ion pressure").withDefault<bool>(false);

  
  // bool Vort_mag, Vort_parcurrent, Vort_polarcurrent, Vort_collision, Vort_parviscous;
  // bool Vort_anomalous;

  Vort_mag = optvort["Vort_mag"].doc("Use magnetic effects in vorticity").withDefault<bool>(false);
  Vort_parcurrent = optvort["Vort_parcurrent"].doc("Use parallel current effects in vorticity").withDefault<bool>(false);
  Vort_polarcurrent = optvort["Vort_polarcurrent"].doc("Use polarization current in vorticity").withDefault<bool>(false);
  Vort_collision = optvort["Vort_collision"].doc("Include collisional effects in vorticity").withDefault<bool>(false);
  Vort_parviscous = optvort["Vort_parviscous"].doc("Include parallel viscous effects in vorticity").withDefault<bool>(false);
  Vort_anomalous = optvort["Vort_anomalous"].doc("Include anomalous effects in vorticity").withDefault<bool>(false);
  Vort_hyper = optvort["Vort_hyper"].doc("Use hyperdiffusion in vorticity").withDefault<bool>(false);
  Vort_numdiff = optvort["Vort_numdiff"].doc("Use parallel numerical diffusion in vorticity").withDefault<bool>(false);

  
  // bool VePsi_parefield, VePsi_parpressure, VePsi_partemp, VePsi_parcurrent, VePsi_ExB, VePsi_parflow;

  VePsi_parefield = optvepsi["VePsi_parefield"].doc("Use parallel electric field in electron velocity").withDefault<bool>(false);
  VePsi_parpressure = optvepsi["VePsi_parpressure"].doc("Use parallel pressure gradient in electron velocity").withDefault<bool>(false);
  VePsi_partemp = optvepsi["VePsi_partemp"].doc("Use parallel temperature gradient in electron velocity").withDefault<bool>(false);
  VePsi_parcurrent = optvepsi["VePsi_parcurrent"].doc("Use parallel current in electron velocity").withDefault<bool>(false);
  VePsi_ExB = optvepsi["VePsi_ExB"].doc("Use ExB drift in electron velocity").withDefault<bool>(false);
  VePsi_parflow = optvepsi["VePsi_parflow"].doc("Use parallel flow effects in electron velocity").withDefault<bool>(false);
  VePsi_hyper = optvepsi["VePsi_hyper"].doc("Use hyperdiffusion in electron velocity").withDefault<bool>(false);
  VePsi_numdiff = optvepsi["VePsi_numdiff"].doc("Use parallel numerical diffusion in electron velocity").withDefault<bool>(false);

  // Initialize the corresponding fields

  TE_Ne = optsc["TE_Ne"].doc("Save all terms in time evolution of density").withDefault<bool>(false);
  TE_NVi = optsc["TE_NVi"].doc("Save all terms in time evolution of ion momentum").withDefault<bool>(false);
  TE_Pe = optsc["TE_Pe"].doc("Save all terms in time evolution of electron pressure").withDefault<bool>(false);
  TE_Pi = optsc["TE_Pi"].doc("Save all terms in time evolution of ion pressure").withDefault<bool>(false);
  TE_Vort = optsc["TE_Vort"].doc("Save all terms in time evolution of vorticity").withDefault<bool>(false);
  TE_VePsi = optsc["TE_VePsi"].doc("Save all terms in time evolution of electron velocity").withDefault<bool>(false);
  
  // Density
  TE_Ne_ExB = 0.0;
  TE_Ne_mag = 0.0;
  TE_Ne_parflow = 0.0;
  TE_Ne_collision = 0.0;
  TE_Ne_anomalous = 0.0;
  TE_Ne_sources = 0.0;
  TE_Ne_hyper = 0.0;
  TE_Ne_numdiff = 0.0;
  if (TE_Ne) {
    SAVE_REPEAT(TE_Ne_ExB, TE_Ne_mag, TE_Ne_parflow, TE_Ne_collision, TE_Ne_anomalous, TE_Ne_sources, TE_Ne_hyper, TE_Ne_numdiff);
  }


  // Ion momentum
  TE_NVi_ExB = 0.0;
  TE_NVi_mag = 0.0;
  TE_NVi_parflow = 0.0;
  TE_NVi_parpressure = 0.0;
  TE_NVi_parviscos = 0.0;
  TE_NVi_collision = 0.0;
  TE_NVi_anomalous = 0.0;
  TE_NVi_hyper = 0.0;
  TE_NVi_numdiff = 0.0;
  if (TE_NVi) {
    SAVE_REPEAT(TE_NVi_ExB, TE_NVi_mag, TE_NVi_parflow, TE_NVi_parpressure, TE_NVi_parviscos, TE_NVi_collision, TE_NVi_anomalous);
    SAVE_REPEAT(TE_NVi_hyper, TE_NVi_numdiff);
  }

  
  // Electron pressure
  TE_Pe_ExB = 0.0;
  TE_Pe_mag = 0.0;
  TE_Pe_parflow = 0.0;
  TE_Pe_conduction = 0.0;
  TE_Pe_ohmic = 0.0;
  TE_Pe_thermalforce = 0.0;
  TE_Pe_thermalcurrent = 0.0;
  TE_Pe_collision = 0.0;
  TE_Pe_anomalous = 0.0;
  TE_Pe_sources = 0.0;
  TE_Pe_energyexchange = 0.0;
  TE_Pe_hyper = 0.0;
  TE_Pe_numdiff = 0.0;
  if (TE_Pe) {
    SAVE_REPEAT(TE_Pe_ExB, TE_Pe_mag, TE_Pe_parflow, TE_Pe_conduction, TE_Pe_ohmic, TE_Pe_thermalforce, TE_Pe_thermalcurrent);
    SAVE_REPEAT(TE_Pe_collision, TE_Pe_anomalous, TE_Pe_sources, TE_Pe_energyexchange, TE_Pe_hyper, TE_Pe_numdiff);
  }
  

  // Ion pressure
  TE_Pi_ExB = 0.0;
  TE_Pi_mag = 0.0;
  TE_Pi_parflow = 0.0;
  TE_Pi_conduction = 0.0;
  TE_Pi_diamagenergyexchange = 0.0;
  TE_Pi_parviscousheat = 0.0;
  TE_Pi_resistivedrift = 0.0;
  TE_Pi_perpviscous = 0.0;
  TE_Pi_sources = 0.0;
  TE_Pi_hyper = 0.0;
  TE_Pi_numdiff = 0.0;
  if (TE_Pi) {
    SAVE_REPEAT(TE_Pi_ExB, TE_Pi_mag, TE_Pi_parflow, TE_Pi_conduction, TE_Pi_diamagenergyexchange, TE_Pi_parviscousheat);
    SAVE_REPEAT(TE_Pi_resistivedrift, TE_Pi_perpviscous, TE_Pi_sources, TE_Pi_hyper, TE_Pi_numdiff);
  }


  // Vorticity
  TE_Vort_mag = 0.0;
  TE_Vort_parcurrent = 0.0;
  TE_Vort_polarcurrent = 0.0;
  TE_Vort_collision = 0.0;
  TE_Vort_parviscous = 0.0;
  TE_Vort_anomalous = 0.0;
  TE_Vort_hyper = 0.0;
  TE_Vort_numdiff = 0.0;
  if (TE_Vort) {
    SAVE_REPEAT(TE_Vort_mag, TE_Vort_parcurrent, TE_Vort_polarcurrent, TE_Vort_collision, TE_Vort_parviscous, TE_Vort_anomalous);
    SAVE_REPEAT(TE_Vort_hyper, TE_Vort_numdiff);
  }


  // Electron velocity
  TE_VePsi_parefield = 0.0;
  TE_VePsi_parpressure = 0.0;
  TE_VePsi_partemp = 0.0;
  TE_VePsi_parcurrent = 0.0;
  TE_VePsi_ExB = 0.0;
  TE_VePsi_parflow = 0.0;
  TE_VePsi_hyper = 0.0;
  TE_VePsi_numdiff = 0.0;
  if (TE_VePsi) {
    SAVE_REPEAT(TE_VePsi_parefield, TE_VePsi_parpressure, TE_VePsi_partemp, TE_VePsi_parcurrent, TE_VePsi_ExB, TE_VePsi_parflow);
    SAVE_REPEAT(TE_VePsi_hyper, TE_VePsi_numdiff);
  }


  
  /////////////////////////////////////////////////////////////////////////

  // Switches to change between different calculation methods

  OPTION(optnumerics, use_Div_n_bxGrad_f_B_XPPM, true);
  OPTION(optnumerics, use_bracket, true);
  OPTION(optnumerics, ne_bndry_flux, false);
  OPTION(optnumerics, pe_bndry_flux, false);
  OPTION(optnumerics, vort_bndry_flux, false);

  OPTION(optsc, boussinesq, false);
  
  // Switches for different methods to support numerical stability
  
  OPTION(optnumerics, floor_kappa_ipar, -1.0);
  OPTION(optnumerics, floor_kappa_epar,-1.0);
  OPTION(optnumerics, NVi_supsonic_dissipation, false);
  OPTION(optnumerics, NVi_supsonic_factor, 1.0);
  OPTION(optnumerics, Ve_supsonic_dissipation, false);
  OPTION(optnumerics, Ve_supsonic_factor, 1.0);

  OPTION(optnumerics, ne_bndry_flux, false);
  OPTION(optnumerics, pe_bndry_flux, false);
  OPTION(optnumerics, vort_bndry_flux, false);

  OPTION(optnumerics, flux_limit_alpha, -1);
  OPTION(optnumerics, kappa_limit_alpha, -1);
  OPTION(optnumerics, eta_limit_alpha, -1);
  
  OPTION(optnumerics, resistivity_multiply, 1.0);
  OPTION(optnumerics, electron_weight, 1.0);
  
  // Sheath switches
  
  OPTION(optsheath, sheath_model, 0);
  OPTION(optsheath, sheath_gamma_e, 5.5);
  OPTION(optsheath, sheath_gamma_i, 1.0);

  OPTION(optsheath, neutral_vwall, 1. / 3);  // 1/3rd Franck-Condon energy at wall
  OPTION(optsheath, sheath_yup, true);       // Apply sheath at yup?
  OPTION(optsheath, sheath_ydown, true);     // Apply sheath at ydown?
  OPTION(optsheath, test_boundaries, false); // Test boundary conditions
  OPTION(optsheath, parallel_sheaths, false); // Apply parallel sheath conditions?
  OPTION(optsheath, par_sheath_model, 0);
  OPTION(optsheath, par_sheath_ve, true)
  OPTION(optsheath, Div_parP_n_sheath_extra, Div_parP_n_sheath_extra);
  sheath_allow_supersonic = optsheath["sheath_allow_supersonic"]
          .doc("If plasma is faster than sound speed, go to plasma velocity")
          .withDefault<bool>(true);

  
  
  // Output additional information
  OPTION(optsc, verbose, false);    // Save additional fields
  OPTION(optsc, output_ddt, false); // Save time derivatives

  // Normalisation
  OPTION(optsc, Tnorm, 20);  // Reference temperature [eV]
  OPTION(optsc, Nnorm, 1e19); // Reference density [m^-3]
  OPTION(optsc, Bnorm, 1.0);  // Reference magnetic field [T]
  OPTION(optsc, AA, 2.0); // Ion mass (2 = Deuterium)
  output.write("Normalisation Te={:e}, Ne={:e}, B={:e}\n", Tnorm, Nnorm, Bnorm);
  SAVE_ONCE(Tnorm, Nnorm, Bnorm, AA); // Save
  Cs0 = sqrt(qe * Tnorm / (AA * Mp)); // Reference sound speed [m/s]
  Omega_ci = qe * Bnorm / (AA * Mp);  // Ion cyclotron frequency [1/s]
  rho_s0 = Cs0 / Omega_ci;
  mi_me = AA * Mp / (electron_weight * Me);
  me_mi = (electron_weight * Me) / (AA * Mp);
  beta_e = qe * Tnorm * Nnorm / (SQ(Bnorm) / (2. * SI::mu0));
  output.write("\tmi_me={}, beta_e={}\n", mi_me, beta_e);
  SAVE_ONCE(mi_me, beta_e, me_mi);
  output.write("\t Cs={:e}, rho_s={:e}, Omega_ci={:e}\n", Cs0, rho_s0, Omega_ci);
  SAVE_ONCE(Cs0, rho_s0, Omega_ci);
  // Collision times
  BoutReal lambda_ei = 24. - log(sqrt(Nnorm / 1e6) / Tnorm);
  BoutReal lambda_ii = 23. - log(sqrt(2. * Nnorm / 1e6) / pow(Tnorm, 1.5));
  tau_e0 = 1. / (2.91e-6 * (Nnorm / 1e6) * lambda_ei * pow(Tnorm, -3. / 2));
  tau_i0 =
      sqrt(AA) / (4.78e-8 * (Nnorm / 1e6) * lambda_ii * pow(Tnorm, -3. / 2));
  output.write("\ttau_e0={:e}, tau_i0={:e}\n", tau_e0, tau_i0);


  // Get the transport parameters
  anomalous_D = opttransport["anomalous_D"].doc("Anomalous diffusion").withDefault(Field3D{0.0});
  anomalous_nu = opttransport["anomalous_nu"].doc("Anomalous viscosity").withDefault(Field3D{0.0});
  anomalous_chi = opttransport["anomalous_chi"].doc("Anomalous condoctivity").withDefault(Field3D{0.0});

  if (anomalous_D > 0.0) {
    // Normalise
    anomalous_D /= rho_s0 * rho_s0 * Omega_ci; // m^2/s
    output.write("\tnormalised anomalous D_perp = {:e}\n", anomalous_D);
    a_d3d = anomalous_D;
    mesh->communicate(a_d3d);
    a_d3d.yup() = anomalous_D;
    a_d3d.ydown() = anomalous_D;
  }

  
  
  if (anomalous_chi > 0.0) {
    // Normalise
    anomalous_chi /= rho_s0 * rho_s0 * Omega_ci; // m^2/s
    output.write("\tnormalised anomalous chi_perp = {:e}\n", anomalous_chi);
    a_chi3d = anomalous_chi;
    mesh->communicate(a_chi3d);
    a_chi3d.yup() = anomalous_D;
    a_chi3d.ydown() = anomalous_D;
  }
  if (anomalous_nu > 0.0) {
    // Normalise
    anomalous_nu /= rho_s0 * rho_s0 * Omega_ci; // m^2/s
    output.write("\tnormalised anomalous nu_perp = {:e}\n", anomalous_nu);
    a_nu3d = anomalous_nu;
    mesh->communicate(a_nu3d);
    a_nu3d.yup() = anomalous_D;
    a_nu3d.ydown() = anomalous_D;
  }

  

  
  FieldFactory fact(mesh);

  
  // Get switches from each variable section

  NeSource = optne["source"].doc("Source term in ddt(Ne)").withDefault(Field3D{0.0});
  NeSource /= Omega_ci;
  Sn = NeSource;

  
  PeSource = optpe["source"].withDefault(Field3D{0.0});
  PeSource /= Omega_ci;
  Spe = PeSource;

  PiSource = optpi["source"].withDefault(Field3D{0.0});
  PiSource /= Omega_ci;
  Spi = PiSource;

  SAVE_ONCE(Sn, Spe, Spi);
  

  /////////////////////////////////////////////////////////
  // Load metric tensor from the mesh, passing length and B
  // field normalisations
  Coordinates *coord = mesh->getCoordinates();
  coord->Bxy /= Bnorm;
  //CONTRAVARIANT

  mul_all_inp(coord->g11, rho_s0 * rho_s0);
  mul_all_inp(coord->g22, rho_s0 * rho_s0);
  mul_all_inp(coord->g33, rho_s0 * rho_s0);
  mul_all_inp(coord->g12, rho_s0 * rho_s0);
  mul_all_inp(coord->g13, rho_s0 * rho_s0);
  mul_all_inp(coord->g23, rho_s0 * rho_s0);


  //Jacobi matrix
  div_all_inp(coord->J, rho_s0 * rho_s0 * rho_s0);

  //LIKE IN D'haeseleer

  //subscripts = ()_i -> covariant

  //superscripts = ()^j -> contravariant

  //COVARIANT
  div_all_inp(coord->g_11, rho_s0 * rho_s0);
  div_all_inp(coord->g_22, rho_s0 * rho_s0); // In m^2
  div_all_inp(coord->g_33, rho_s0 * rho_s0);
  div_all_inp(coord->g_12, rho_s0 * rho_s0);
  div_all_inp(coord->g_13, rho_s0 * rho_s0);
  div_all_inp(coord->g_23, rho_s0 * rho_s0);

  coord->geometry(); // Calculate other metrics

  _FCIDiv_a_Grad_perp = std::make_unique<FCI::dagp_fv>(*mesh);
  *_FCIDiv_a_Grad_perp *= rho_s0;

  if (Options::root()["mesh:paralleltransform"]["type"].as<std::string>() == "fci") {
    fci_transform = true;
  }else{
    fci_transform = false;
  }
  ASSERT0(fci_transform);

  if(fci_transform){
    mesh->get(Bxyz, "B",1.0);
    mesh->get(coord->Bxy, "Bxy", 1.0);
    Bxyz /= Bnorm;
    coord->Bxy /= Bnorm;
    // mesh->communicate(Bxyz, coord->Bxy); // To get yup/ydown fields
    //  Note: A Neumann condition simplifies boundary conditions on fluxes
    //  where the condition e.g. on J should be on flux (J/B)

    auto logBxy = log(coord->Bxy);
    auto logBxyz = log(Bxyz);
    logBxy.applyBoundary("neumann");
    logBxyz.applyBoundary("neumann");
    mesh->communicate(logBxy, logBxyz);
    logBxy.applyParallelBoundary(parbc);
    logBxyz.applyParallelBoundary(parbc);
    output_info.write("Setting from log");
    coord->Bxy = exp_all(logBxy);
    Bxyz = exp_all(logBxyz);
    SAVE_ONCE(Bxyz);
    ASSERT1(min(Bxyz) > 0.0);
    fwd_bndry_mask = BoutMask(mesh, false);
    bwd_bndry_mask = BoutMask(mesh, false);
    for (const auto &bndry_par : mesh->getBoundariesPar(BoundaryParType::fwd)) {
      for (const auto &pnt : *bndry_par) {
	fwd_bndry_mask[pnt.ind()] = true;
      }
    }
    for (const auto &bndry_par : mesh->getBoundariesPar(BoundaryParType::bwd)) {
      for (const auto &pnt : *bndry_par) {
        bwd_bndry_mask[pnt.ind()] = true;
      }
    }

    bout::checkPositive(coord->Bxy, "f", "RGN_NOCORNERS");
    bout::checkPositive(coord->Bxy.yup(), "fyup", "RGN_YPAR_+1");
    bout::checkPositive(coord->Bxy.ydown(), "fdown", "RGN_YPAR_-1");
    logB = log(Bxyz);

    bracket_factor = sqrt(coord->g_22) / (coord->J * Bxyz);
    SAVE_ONCE(bracket_factor);
  }else{
    mesh->communicate(coord->Bxy);
    bracket_factor = sqrt(coord->g_22) / (coord->J * coord->Bxy);
    SAVE_ONCE(bracket_factor);
  }

  B12 = sqrt_all(coord->Bxy);     // B^(1/2)
  B32 = mul_all(B12, coord->Bxy); // B^(3/2)
  B42 = SQ_all(coord->Bxy);



  /////////////////////////////////////////////////////////
  // Read profiles from the mesh
  TRACE("Reading profiles");

  Field3D NeMesh, TeMesh, TiMesh;
  if (mesh->get(NeMesh, "Ne0")) {
    // No Ne0. Try Ni0
    if (mesh->get(NeMesh, "Ni0")) {
      output << "WARNING: Neither Ne0 nor Ni0 found in mesh input\n";
    }
  }
  NeMesh *= 1e20; // Convert to m^-3

  NeMesh /= Nnorm; // Normalise

  if (mesh->get(TeMesh, "Te0")) {
    // No Te0
    output << "WARNING: Te0 not found in mesh\n";
    // Try to read Ti0
    if (mesh->get(TeMesh, "Ti0")) {
      // No Ti0 either
      output << "WARNING: No Te0 or Ti0. Setting TeMesh to 0.0\n";
      TeMesh = 0.0;
    }
  }

  TeMesh /= Tnorm; // Normalise

  if (mesh->get(TiMesh, "Ti0")) {
    // No Ti0
    output << "WARNING: Ti0 not found in mesh. Setting to TeMesh\n";
    TiMesh = TeMesh;
  }
  TiMesh /= Tnorm; // Normalise
  PiTarget = NeMesh * TiMesh;

  NeTarget = NeMesh;
  PeTarget = NeMesh * TeMesh;

  if (!restarting && !ramp_mesh) {
    if (optsc["startprofiles"].withDefault(true)) {
      Ne += NeMesh; // Add profiles in the mesh file

      Pe += NeMesh * TeMesh;

      Pi += NeMesh * TiMesh;
      // Check for negatives
      if (min(Pi, true) < 0.0) {
        throw BoutException("Starting ion pressure is negative");
      }
      if (max(Pi, true) < 1e-5) {
        throw BoutException("Starting ion pressure is too small");
      }
      mesh->communicateXZ(Pi);
    }

    // Check for negatives
    if (min(Ne, true) < 0.0) {
      throw BoutException("Starting density is negative");
    }
    if (max(Ne, true) < 1e-5) {
      throw BoutException("Starting density is too small");
    }

    if (min(Pe, true) < 0.0) {
      throw BoutException("Starting pressure is negative");
    }
    if (max(Pe, true) < 1e-5) {
      throw BoutException("Starting pressure is too small");
    }

    mesh->communicateXZ(Ne, Pe);
  }

  /////////////////////////////////////////////////////////
  // Read curvature components
  TRACE("Reading curvature");

  try {
    Curlb_B.covariant = false; // Contravariant
    mesh->get(Curlb_B, "bxcv");
    // SAVE_ONCE(Curlb_B);
  } catch (BoutException &e) {
    try {
      // May be 2D, reading as 3D
      Vector2D curv2d;
      curv2d.covariant = false;
      mesh->get(curv2d, "bxcv");
      Curlb_B = curv2d;
    } catch (BoutException &e) {
      if (j_diamag) {
        // Need curvature
        throw;
      } else {
        output_warn.write("No curvature vector in input grid");
        Curlb_B = 0.0;
      }
    }
  }

  if (!use_bracket){
    TRACE("Reading curvature for the curvature drifts");
    try{
      mesh->get(bxcvx,"bxcvx");
      mesh->get(bxcvy,"bxcvy");
      mesh->get(bxcvz,"bxcvz");

      //Normalize

      bxcvx /= Bnorm;
      bxcvy /= Bnorm;
      bxcvz /= Bnorm;
      /*
      bxcvx *= rho_s0;
      bxcvy *= rho_s0;
      bxcvz *= rho_s0;
      */
      
      bxcv = 0.0;
      bxcv.covariant = false;
      bxcv.x = bxcvx;
      bxcv.y = bxcvy;
      bxcv.z = bxcvz;
      bxcv.covariant = false;
      SAVE_ONCE(bxcvx,bxcvy,bxcvz);
      
    } catch(BoutException &e) {
      throw;
    }
  }



  //////////////////////////////////////////////////////////////
  // Electromagnetic fields

  opt["phiSolver"].setConditionallyUsed();
  optsc["newXZsolver"].setConditionallyUsed();

  OPTION(optsc, newXZsolver, false);
  if (newXZsolver) {
    // Test new LaplaceXZ solver                                                                                                                  
    newSolver = LaplaceXZ::create(bout::globals::mesh);
    // Set coefficients for Boussinesq solve                                                                                                      
    newSolver->setCoefs(1. / SQ(coord->Bxy), Field3D(0.0));
  } else {
    // Use older Laplacian solver                                                                                                                 
    phiSolver = Laplacian::create(&opt["phiSolver"]);
    // Set coefficients for Boussinesq solve                                                                                                      
    phiSolver->setCoefC(1./ SQ(coord->Bxy));
  }
  phi = 0.0;
  phi.setBoundary("phi"); // For y boundaries                                                                                                     

      // Add phi to restart files so that the value in the boundaries                                                                                 
      // is restored on restart. This is done even when phi is not evolving,                                                                          
      // so that phi can be saved and re-loaded                                                                                                       
  restart.addOnce(phi, "phi");
  aparSolver = Laplacian::create(&opt["aparSolver"]);
  Ve.setBoundary("Ve");
  nu.setBoundary("nu");
  Jpar.setBoundary("Jpar");
  psi = 0.0;
  

  
  nu = 0.0;
  kappa_epar = 0.0;
  kappa_ipar = 0.0;
  Dn = 0.0;
  debug_visheath = 0.0;
  debug_vesheath = 0.0;
  debug_sheathexp = 0.0;
  debug_soundspeed = 0.0;
  debug_VePsisheath = 0.0;
  debug_phisheath = 0.0;
  debug_denom = 0.0;
  debug_phibndry3d = 0.0;
  NVi_dampening = 0.0;
  Ve_dampening = 0.0;
  
  
  SAVE_REPEAT(a,b,d);
  SAVE_REPEAT(Te, Ti);
  NVi_Div_parP_n = 0.0;
  if (verbose) {
    // Save additional fields
    SAVE_REPEAT(Jpar); // Parallel current
    SAVE_REPEAT(debug_soundspeed,debug_phibndry3d);
    SAVE_REPEAT(tau_e, tau_i);

    if(NVi_supsonic_dissipation){
      SAVE_REPEAT(NVi_dampening);
    }
    
    if(Ve_supsonic_dissipation){
      SAVE_REPEAT(Ve_dampening);
    }
    
    if(kappa_limit_alpha>0.0){
      SAVE_REPEAT(debug_denom);
    }
    
    SAVE_REPEAT(kappa_epar); // Parallel electron heat conductivity
    SAVE_REPEAT(kappa_ipar); // Parallel ion heat conductivity
    SAVE_REPEAT(nu);
    SAVE_REPEAT(debug_visheath,debug_vesheath,debug_sheathexp);
    SAVE_REPEAT(NVi_Div_parP_n);
    SAVE_REPEAT(debug_phisheath);
    SAVE_REPEAT(debug_VePsisheath);

  }

  zero_all(phi);
  zero_all(psi);

  // Preconditioner
  setPrecon((preconfunc)&Hermes::precon);

  
  if (evolve_te && parallel_sheaths){
    SAVE_REPEAT(sheath_dpe);
  }

  if (evolve_ti && parallel_sheaths){
    SAVE_REPEAT(sheath_dpi);
  }
  zero_all(Ve);
  
  // Magnetic field in boundary
  auto& Bxy = mesh->getCoordinates()->Bxy;
  
  opt["Pn"].setConditionallyUsed();
  opt["Nn"].setConditionallyUsed();
  opt["NVn"].setConditionallyUsed();
  opt["Pe"].setConditionallyUsed();
  opt["Pi"].setConditionallyUsed();
  opt["Vn"].setConditionallyUsed();
  opt["Vn_x"].setConditionallyUsed();
  opt["Vn_y"].setConditionallyUsed();
  opt["Vn_z"].setConditionallyUsed();
  opt["phi"].setConditionallyUsed();
  opt["phiSolver"].setConditionallyUsed();
  opt["Vort"].setConditionallyUsed();
  opt["VePsi"].setConditionallyUsed();
  optsc["neutral_gamma"].setConditionallyUsed();

  alloc_all(Te);
  alloc_all(Ti);
  alloc_all(Vi);
  alloc_all(a);
  alloc_all(b);
  alloc_all(d);
  return 0;
}

int Hermes::rhs(BoutReal t) {
  if (show_timesteps) {
    printf("TIME = %e\r", t);
  }

  if (!evolve_plasma) {
    Ne = 0.0;
    Pe = 0.0;
    Pi = 0.0;
    Vort = 0.0;
    VePsi = 0.0;
    NVi = 0.0;
    sheath_model = 0;
  }

  Coordinates *coord = mesh->getCoordinates();
  
  // Communicate evolving variables
  // Note: Parallel slices are not calculated because parallel derivatives
  // are calculated using field aligned quantities
  mesh->communicate(EvolvingVars);
  Ne.applyParallelBoundary();
  Vort.applyParallelBoundary();
  if (evolve_te){
    Pe.applyParallelBoundary();
  }
  if (evolve_ti){
    Pi.applyParallelBoundary();
  }
  NVi.applyParallelBoundary();
  
  if (evolve_VePsi){
    VePsi.applyParallelBoundary();
  }

  bool currents = j_par | j_diamag;

  Field3D sound_speed;
  sound_speed.allocate();

  alloc_all(Te);
  alloc_all(Ti);
  alloc_all(Vi);
  alloc_all(Pi);
  alloc_all(Pe);

  BOUT_FOR(i, Ne.getRegion("RGN_ALL")) {
    // Field3D Ne = floor_all(Ne, 1e-5);
    floor_all(Ne, 1e-2, i);

    if (!evolve_te) {
      copy_all(Pe, Ne, i); // Fixed electron temperature
    }

    div_all(Te, Pe, Ne, i);
    // ASSERT0(Te[i] > 1e-10);
    /// printf("%f\n", Te[i]);
    div_all(Vi, NVi, Ne, i);

    floor_all(Te, 0.05, i);
    // ASSERT0(Te[i] > 1e-10);
    
    mul_all(Pe, Te, Ne, i);

    if (!evolve_ti) {
      copy_all(Pi, Ne, i); // Fixed ion temperature
    }

    div_all(Ti, Pi, Ne, i);
    floor_all(Ti, 0.05, i);
    mul_all(Pi, Ti, Ne, i);
    div_all(Te, Pe, Ne, i);
    // ASSERT0(Te[i] > 1e-10);

    sound_speed[i] = scale_num_cs * sqrt(Te[i] + Ti[i] * (5. / 3));
  }

  sound_speed.applyBoundary("neumann");
  
  if(verbose){
    debug_soundspeed = sound_speed;
  }
  
  // Set radial boundary conditions on Te, Ti, Vi
  //
  if (mesh->firstX()) {
    for (int j = mesh->ystart; j <= mesh->yend; j++) {
      for (int k = 0; k < mesh->LocalNz; k++) {
        BoutReal ne_bndry = 0.5 * (Ne(1, j, k) + Ne(2, j, k));
        if (ne_bndry < 1e-2)
          ne_bndry = 1e-2;
        BoutReal pe_bndry = 0.5 * (Pe(1, j, k) + Pe(2, j, k));
        BoutReal pi_bndry = 0.5 * (Pi(1, j, k) + Pi(2, j, k));

        BoutReal te_bndry = pe_bndry / ne_bndry;
        BoutReal ti_bndry = pi_bndry / ne_bndry;

        Te(1, j, k) = 2. * te_bndry - Te(2, j, k);
        Ti(1, j, k) = 2. * ti_bndry - Ti(2, j, k);
        Vi(0, j, k) = Vi(1, j, k) = Vi(2, j, k);

        if (te_bndry < 0.1 / Tnorm)
          te_bndry = 0.1 / Tnorm;
        if (ti_bndry < 0.1 / Tnorm)
          ti_bndry = 0.1 / Tnorm;

        Te(1, j, k) = 2. * te_bndry - Te(2, j, k);
        Ti(1, j, k) = 2. * ti_bndry - Ti(2, j, k);
      }
    }
  }
  if (mesh->lastX()) {
    int n = mesh->LocalNx;
    for (int j = mesh->ystart; j <= mesh->yend; j++) {
      for (int k = 0; k < mesh->LocalNz; k++) {
        BoutReal ne_bndry = 0.5 * (Ne(n - 1, j, k) + Ne(n - 2, j, k));
        if (ne_bndry < 1e-2)
          ne_bndry = 1e-2;
        BoutReal pe_bndry = 0.5 * (Pe(n - 1, j, k) + Pe(n - 2, j, k));
        BoutReal pi_bndry = 0.5 * (Pi(n - 1, j, k) + Pi(n - 2, j, k));

        BoutReal te_bndry = pe_bndry / ne_bndry;
        BoutReal ti_bndry = pi_bndry / ne_bndry;

        Te(n - 1, j, k) = 2. * te_bndry - Te(n - 2, j, k);
        Ti(n - 1, j, k) = 2. * ti_bndry - Ti(n - 2, j, k);
        Vi(n - 1, j, k) = Vi(n - 2, j, k);

        if (te_bndry < 0.05)
          te_bndry = 0.05;
        if (ti_bndry < 0.05)
          ti_bndry = 0.05;

        Te(n - 1, j, k) = 2. * te_bndry - Te(n - 2, j, k);
        Ti(n - 1, j, k) = 2. * ti_bndry - Ti(n - 2, j, k);
      }
    }
  }

  /////////////////////////////////////////////////////////////
  // Calculate additional variables that are used for various calculations

  Field3D Te32= pow(Te,1.5);
  Te32.applyBoundary("neumann");
  mesh->communicate(Te32);
  Te32.applyParallelBoundary(parbc);
  
  //////////////////////////////////////////////////////////////
  // Calculate electrostatic potential phi
 
  TRACE("Electrostatic potential");
  Field3D phi_boundary3d;
  phi_boundary3d = 3.0 * Te;
  

    
  if (boussinesq) {
		
    if (mesh->firstX()) {
      for (int j = mesh->ystart; j <= mesh->yend; j++) {
	for (int k = 0; k < mesh->LocalNz; k++) {
	  // Average phi + Pi at the boundary, and set the boundary cell
	  // to this value. The phi solver will then put the value back
	  // onto the cell mid-point
	  phi_boundary3d(mesh->xstart - 1, j, k) =
	    0.5
	    * (phi_boundary3d(mesh->xstart - 1, j, k) +
	       phi_boundary3d(mesh->xstart, j, k) +
	       Pi(mesh->xstart - 1, j, k) +
	       Pi(mesh->xstart, j, k));
	}
      }
    }

    if (mesh->lastX()) {
      for (int j = mesh->ystart; j <= mesh->yend; j++) {
	for (int k = 0; k < mesh->LocalNz; k++) {
	  phi_boundary3d(mesh->xend + 1, j, k) =
	    0.5
	    * (phi_boundary3d(mesh->xend + 1, j, k) +
	       phi_boundary3d(mesh->xend, j, k) +
	       Pi(mesh->xend + 1, j, k) +
	       Pi(mesh->xend, j, k));
	}
      }
    }
        
    ////////////////////////////////////////////
    // Boussinesq, non-split
    // Solve all components using X-Z solver
    
    if (newXZsolver) {
      // Use the new LaplaceXZ solver
      // newSolver->setCoefs(1./SQ(coord->Bxy), 0.0); // Set when initialised
      phi = newSolver->solve(Vort, phi + Pi);
    } else {
      // Use older Laplacian solver
      // phiSolver->setCoefC(1./SQ(coord->Bxy)); // Set when initialised
      mesh->communicate(phi_boundary3d);
      phi = phiSolver->solve(mul_all(Vort , mul_all(coord->Bxy, coord->Bxy)), phi_boundary3d);//_boundary3d);
      //phi = phiSolver->solve(Vort, phi);
    }
        
    // Hot ion term in vorticity
    debug_phibndry3d = phi_boundary3d;
    mesh->communicate(phi);
    phi.applyParallelBoundary(parbc);
    phi = sub_all(phi, Pi);
	
  } else {
    ////////////////////////////////////////////
    // Non-Boussinesq
    //
    throw BoutException("Non-Boussinesq not implemented yet");
  }
  


  
  //////////////////////////////////////////////////////////////
  TRACE("Calculating resistivity");
  tau_e = div_all(mul_all(mul_all(div_all(Cs0 , rho_s0) , tau_e0) , Te32) , Ne);
  nu = resistivity_multiply / (1.96 * tau_e * mi_me);
  nu.applyBoundary("neumann");
  mesh->communicate(nu);
  nu.applyParallelBoundary(parbc);

  

  
  //////////////////////////////////////////////////////////////
  // Calculate perturbed magnetic field psi
  TRACE("Calculating psi");

  
  if (electromagnetic) {
    if (FiniteElMass) {
      // Solve Helmholtz equation for psi
      
      aparSolver->setCoefA(-Ne*0.5*mi_me*beta_e);
      aparSolver->setCoefC(Field3D(1.0));
      
      psi = aparSolver->solve(Field3D(-Ne * VePsi), Field3D(psi));
      mesh->communicate(psi);
      psi.applyParallelBoundary(parbc);
      
      Ve = VePsi - 0.5 * beta_e * mi_me * psi + Vi;
	
      Ve.applyBoundary(t);
      mesh->communicate(Ve, psi);
      Ve.applyParallelBoundary(parbc);
      
      Jpar = mul_all(Ne, sub_all(Vi, Ve));
      mesh->communicate(Jpar);
      Jpar.applyParallelBoundary(parbc);

    } else {
      throw BoutException("Running without finite electron mass is not possible anymore!");
    }
  } else {
    // Electrostatic
    zero_all(psi);
    // No psi contribution to VePsi
    Ve = add_all(VePsi , Vi);
  }

  //////////////////////////////////////////////////////////////
  // Sheath boundary conditions on Y up and Y down
  
  TRACE("Sheath boundaries");
  if (parallel_sheaths){
    switch (par_sheath_model) {
    case 0 :{
      for (const auto &bndry_par :
           mesh->getBoundariesPar(BoundaryParType::xout)) {
        for (const auto &pnt : *bndry_par)  {
          int x = pnt.ind().x();
          int y = pnt.ind().y();
          int z = pnt.ind().z();
	  
          // Zero-gradient density
          BoutReal nesheath = floor(Ne(x, y, z), 0.0);

          // Temperature at the sheath entrance
          BoutReal tesheath = floor(Te(x, y, z), 0.0);
          BoutReal tisheath = floor(Ti(x, y, z), 0.0);

          // Zero-gradient potential
          BoutReal phisheath = phi(x, y, z);
	  if (verbose){
	    debug_phisheath(x,y,z) = phisheath;
	  }
          BoutReal visheath = bndry_par->dir * sqrt(tisheath + tesheath);

	  if (sheath_allow_supersonic) {
            if (bndry_par->dir == 1){
              if (Vi(x, y, z) > visheath){
                // If plasma is faster, go to plasma velocity
                visheath = Vi(x, y, z);
              }
            } else {
              if (Vi(x, y, z) < visheath){
                visheath = Vi(x, y, z);
              }
            }
          }

	  if (verbose){
	    debug_visheath(x,y,z) = visheath;
	  }

	  
          // Sheath current
          // Note that phi/Te >= 0.0 since for phi < 0
          // vesheath is the electron saturation current
          BoutReal phi_te =
            floor(phisheath / tesheath, 0.0);

          BoutReal vesheath =
            bndry_par->dir * sqrt(tesheath) * (sqrt(mi_me) / (2. * sqrt(PI))) * exp(-phi_te);
	  
	  if (verbose){
	    debug_sheathexp(x,y,z) = exp(-phi_te);
	    debug_vesheath(x,y,z) = vesheath;
	  }
          // J = n*(Vi - Ve)
          BoutReal jsheath = nesheath * (visheath - vesheath);
	  BoutReal VePsisheath= vesheath-visheath;
	  if (nesheath < 1e-10) {
            vesheath = visheath;
            jsheath = 0.0;
          }
	  if (verbose){
	    debug_VePsisheath (x,y,z) = VePsisheath;
	  }

          // Neumann conditions
          Ne.ynext(bndry_par->dir)(x, y+bndry_par->dir, z) = nesheath;
          phi.ynext(bndry_par->dir)(x, y+bndry_par->dir, z) = phisheath;
          Vort.ynext(bndry_par->dir)(x, y+bndry_par->dir, z) = Vort(x, y, z);

          // Here zero-gradient Te, heat flux applied later
          Te.ynext(bndry_par->dir)(x, y+bndry_par->dir, z) = Te(x, y, z);
          Ti.ynext(bndry_par->dir)(x, y+bndry_par->dir, z) = Ti(x, y, z);

          Pe.ynext(bndry_par->dir)(x, y+bndry_par->dir, z) = Pe(x, y, z);
          Pi.ynext(bndry_par->dir)(x, y+bndry_par->dir, z) = Pi(x, y, z);

          // Dirichlet conditions

	  if (electromagnetic || FiniteElMass){
	    VePsi.ynext(bndry_par->dir)(x, y+bndry_par->dir, z) = VePsisheath;
	  }

	  
          Vi.ynext(bndry_par->dir)(x, y+bndry_par->dir, z) = visheath;//2. * visheath - Vi(x, y, z);
          if (par_sheath_ve){
            Ve.ynext(bndry_par->dir)(x, y+bndry_par->dir, z) = vesheath;//2. * vesheath - Ve(x, y, z);
          }
          Jpar.ynext(bndry_par->dir)(x, y+bndry_par->dir, z) = jsheath;
            // 2. * jsheath - Jpar(x, y, z);
          NVi.ynext(bndry_par->dir)(x, y+bndry_par->dir, z) = nesheath * visheath;//
            // 1. * nesheath * visheath;// - NVi(x, y, z);
        }
      }
      break;
    }
    case 1: { // insulating boundary      break;
      throw BoutException("Not implemented");
      break;
    }
    default: {
      throw BoutException("Not implemented");
      break;
    }
    }
  }


  //////////////////////////////////////////////////////////////
  // Plasma quantities calculated.
  // At this point we have calculated all boundary conditions,
  // and auxilliary variables like jpar, phi, psi


  TRACE("Collisions");

  const BoutReal tau_e1 = (Cs0 / rho_s0 ) * tau_e0;
  const BoutReal tau_i1 = (Cs0 / rho_s0 ) * tau_i0;
  
  alloc_all(tau_e);
  alloc_all(tau_i);
  BOUT_FOR(i, Te.getRegion("RGN_ALL")) {
    tau_e[i] = tau_e1 * (Te[i] * sqrt(Te[i]) / Ne[i]);
    tau_e.yup()[i] = tau_e1 * (Te.yup()[i] * sqrt(Te.yup()[i]) / Ne.yup()[i]);
    tau_e.ydown()[i] = tau_e1 * (Te.ydown()[i] * sqrt(Te.ydown()[i]) / Ne.ydown()[i]);

    // Normalised ion-ion collision time
    tau_i[i] = tau_i1 * (Ti[i] * sqrt(Ti[i])) / Ne[i];
    tau_i.yup()[i] = tau_i1 * (Ti.yup()[i] * sqrt(Ti.yup()[i])) / Ne.yup()[i];
    tau_i.ydown()[i] = tau_i1 * (Ti.ydown()[i] * sqrt(Ti.ydown()[i])) / Ne.ydown()[i];

  }

  TRACE("Parallel heat conduction");
  
  kappa_epar = mul_all(mul_all(mul_all(mul_all(3.16, mi_me), Te), Ne), tau_e);

  if (kappa_limit_alpha > 0.0) {
    TRACE("electron heat flux limiter");
    /*
     * Flux limiter, as used in SOLPS.
     *
     * Calculate the heat flux from Spitzer-Harm and flux limit
     *
     * Typical value of alpha ~ 0.2 for electrons
     *
     * R.Schneider et al. Contrib. Plasma Phys. 46, No. 1-2, 3 – 191 (2006)
     * DOI 10.1002/ctpp.200610001
     */
    kappa_epar.applyBoundary("neumann");
    mesh->communicate(kappa_epar);
    kappa_epar.applyParallelBoundary(parbc);
      
    Field3D gradTe = Grad_parP(Te);
    gradTe.applyBoundary("neumann");
    mesh->communicate(gradTe);
    gradTe.applyParallelBoundary(parbc);
  
    Field3D q_SH = mul_all(kappa_epar,gradTe);      
    Field3D q_fl = mul_all(kappa_limit_alpha,mul_all(sqrt(mi_me),mul_all(Ne,Te32)));
    Field3D one;
    set_all(one, 1.0);

    Field3D denom = one + abs(div_all(q_SH,q_fl));
    denom.applyBoundary("neumann");
    mesh->communicate(denom);
    denom.applyParallelBoundary(parbc);
    
    if (verbose){
      debug_denom = denom;
    }
      
    kappa_epar = div_all(kappa_epar,denom);
  }

  // Ion parallel heat conduction
  kappa_ipar = mul_all(mul_all(mul_all(3.9, Ti), Ne), tau_i);

  
  
  if(floor_kappa_epar>0.0){
    BOUT_FOR(i, Te.getRegion("RGN_ALL")){
      floor_all(kappa_epar,floor_kappa_epar,i);
    } 
  }

  if(floor_kappa_ipar>0.0){
    BOUT_FOR(i, Te.getRegion("RGN_ALL")){
      floor_all(kappa_ipar,floor_kappa_ipar,i);
    }
  }


  
  ///////////////////////////////////////////////////////////
  // Density
  // This is the electron density equation
  TRACE("density");

  ddt(Ne) = 0.0;

  // FLAGS:bool Ne_ExB, Ne_mag, Ne_parflow, Ne_collision, Ne_anomalous, Ne_sources;

  if (evolve_Ne){

    if (Ne_ExB){
      TRACE("Density ExB");
      
      if (use_Div_n_bxGrad_f_B_XPPM){
	TE_Ne_ExB = -Div_n_bxGrad_f_B_XPPM(Ne, phi, ne_bndry_flux, poloidal_flows,true,bracket_factor) * scale_ExB;
	ddt(Ne) += TE_Ne_ExB;
      } else {
	TE_Ne_ExB = -bracket(phi,Ne, BRACKET_ARAKAWA) * bracket_factor*scale_ExB;
	ddt(Ne) += TE_Ne_ExB;
      }
    }  // End Ne_ExB


    
    if (Ne_mag){
      TRACE("Density mag");
      TE_Ne_mag = -fci_curvature(Pe , use_bracket);
      ddt(Ne) += TE_Ne_mag;
    }  // End Ne_mag


    
    if (Ne_parflow){
      TRACE("Density parflow");
      Field3D neve = mul_all(Ne,Ve);
      TE_Ne_parflow = -Div_par(neve);
      ddt(Ne) += TE_Ne_parflow;
    }  // End Ne_parflow

    if (Ne_collision){
      TRACE("Density collisions");
      throw BoutException("Density collisions not implemented");
    }  // End Ne_collision

    if (Ne_anomalous){
      TRACE("Density anomalous");
      TE_Ne_anomalous = FCIDiv_a_Grad_perp(a_d3d, Ne);
      ddt(Ne) += TE_Ne_anomalous;
    }  // End Ne_anomalous

    if (Ne_sources){
      TRACE("Density sources");
      TE_Ne_sources=NeSource;
      ddt(Ne) += TE_Ne_sources;
    }  // End Ne_sources

    
  }
  
  

  if (bool_ne_hyper) {
    auto tmp = -ne_hyper * ( (SQ(SQ(coord->dz)))  * D4DZ4(Ne) + SQ(SQ(coord->dx))*D4DX4(Ne)  );
    if (TE_Ne){
      TE_Ne_hyper = tmp;
    }
    ddt(Ne) += tmp;
  }
  

  if (bool_numdiff) {
    BOUT_FOR(i, Ne.getRegion("RGN_NOBNDRY")) {
      TE_Ne_numdiff[i] = numdiff[i]*(Ne.ydown()[i.ym()] - 2.*Ne[i] + Ne.yup()[i.yp()]);
    }
    ddt(Ne) += TE_Ne_numdiff;
  }

  ///////////////////////////////////////////////////////////
  // Vorticity
  // This is the current continuity equation

  TRACE("vorticity");

  ddt(Vort) = 0.0;

  if (currents && evolve_vort) {

    if (j_par) {
      TRACE("Vort:j_par");
      vort_jpar = Div_parP(Jpar);
      ddt(Vort) += vort_jpar;
    }

    if (j_diamag) {
      vort_dia = fci_curvature(add_all(Pi , Pe),use_bracket);
      ddt(Vort) += vort_dia;
    }

    // Advection of vorticity by ExB
    if (boussinesq) {
      TRACE("Vort:boussinesq");
      // Using the Boussinesq approximation
      
      if (j_pol_pi){
 
	throw BoutException("j_pol_pi not implemented!");
	
      }else if (j_pol_simplified) {
	// use simplified polarization term from i.e. GBS
	if (use_Div_n_bxGrad_f_B_XPPM){
	  vort_ExB = Div_n_bxGrad_f_B_XPPM(Vort, phi, vort_bndry_flux,
					   poloidal_flows, false , bracket_factor) * scale_ExB;    
	  ddt(Vort) -= vort_ExB;
	} else {
	  vort_ExB = bracket(phi,Vort, BRACKET_ARAKAWA) * bracket_factor * scale_ExB;
	  ddt(Vort) -= vort_ExB;
	}
	  
      }
    } else {
      // When the Boussinesq approximation is not made,
      // then the changing ion density introduces a number
      // of other terms.

      throw BoutException("Hot ion non-Boussinesq not implemented yet\n");
    }
  
    if (anomalous_nu > 0.0) {
      TRACE("Vort:anomalous_nu");
      // Perpendicular anomalous momentum diffusion
      vort_anom = FCIDiv_a_Grad_perp(a_nu3d, Vort);
      ddt(Vort) += vort_anom;
    }

   
    if(bool_Vort_hyper){
      vort_hyper = -Vort_hyper * (SQ(SQ(coord->dz)) * D4DZ4(Vort) + SQ(SQ(coord->dx)) * D4DX4(Vort));
      ddt(Vort) += vort_hyper;
    }
    
    if (bool_numdiff) {
      for(auto &i : NVi.getRegion("RGN_NOBNDRY")) {
        vort_numdiff[i] = numdiff[i]*(Vort.ydown()[i.ym()] - 2.*Vort[i] + Vort.yup()[i.yp()]);
      }
      ddt(Vort) += vort_numdiff;
    }
  }

  ///////////////////////////////////////////////////////////
  // Ohm's law
  // VePsi = Ve - Vi + 0.5*mi_me*beta_e*psi
  TRACE("Ohm's law");

  ddt(VePsi) = 0.0;
  
  if ( electromagnetic || FiniteElMass) {
    // Evolve VePsi except for electrostatic and zero electron mass case
    if (pe_par){
      auto tmp = -mi_me * Grad_parP(Pe) / Ne;
      if(TE_VePsi){
	TE_VePsi_pe_par = tmp;
      }
      ddt(VePsi) += tmp;
    }

    if ( resistivity){
      auto tmp = -mi_me * nu * (Ve - Vi);
      if(TE_VePsi){
	TE_VePsi_resistivity = tmp;
      }
      ddt(VePsi) += tmp;
    }

    if (anomalous_nu>0.0){
      auto tmp = FCIDiv_a_Grad_perp( a_nu3d, VePsi);
      if(TE_VePsi){
	TE_VePsi_anom = tmp;
      }
      ddt(VePsi) += tmp;
    }


    
    // Parallel electric field
    if (j_par) {
      auto tmp = mi_me * Grad_parP(phi);
      if(TE_VePsi){
	TE_VePsi_j_par = tmp;
      }
      ddt(VePsi) += tmp;
    }


    if (thermal_force) {
      auto tmp = -mi_me * 0.71 * Grad_parP(Te);
      if (TE_VePsi){
	TE_VePsi_thermal_force = tmp;
      }
      ddt(VePsi) += tmp;
    }

    
    if (FiniteElMass) {
      // Finite Electron Mass. Small correction needed to conserve energy
      Field3D vdiff = sub_all(Vi,Ve);
      Field3D tmp = 0.0;
      if (Ohmslaw_use_ve){
	tmp = Ve * Grad_par(vdiff);
      } else {
	tmp = Vi * Grad_par(vdiff);
      }

      if (TE_VePsi){
	TE_VePsi_par_adv = tmp;
      }
      ddt(VePsi) += tmp; // Parallel advection
      //ddt(VePsi) -= bracket(phi, vdiff, BRACKET_ARAKAWA)*bracket_factor;  // ExB advection

      if (VePsi_perp){
	// The signs are swapped because vdiff is Vi-Ve and not Ve-Vi 
	if(use_Div_n_bxGrad_f_B_XPPM){
	  TE_VePsi_perp = Div_n_bxGrad_f_B_XPPM(vdiff, phi, false,poloidal_flows , false, bracket_factor) * scale_ExB;
	} else {
	  TE_VePsi_perp = bracket(phi,vdiff,BRACKET_ARAKAWA) * bracket_factor * scale_ExB;
	}
	ddt(VePsi) += TE_VePsi_perp;
      }

      // Should also have ion polarisation advection here
    }

    if (bool_numdiff) {
      for(auto &i : VePsi.getRegion("RGN_NOBNDRY")) {
	auto tmp = numdiff[i]*(VePsi.ydown()[i.ym()] - 2.*VePsi[i] + VePsi.yup()[i.yp()]);
	if(TE_VePsi){
	  TE_VePsi_numdiff[i] = tmp;
	}
        ddt(VePsi)[i] += tmp;
      }
    }

    if (bool_VePsi_hyper){
      auto tmp = -VePsi_hyper*((SQ(SQ(coord->dx)))*D4DX4(VePsi) + (SQ(SQ(coord->dz)))*D4DZ4(VePsi));
      if(TE_VePsi){
	TE_VePsi_hyper = tmp;
      }
      ddt(VePsi) += tmp;
    }

    if(Ve_supsonic_dissipation){
      Field3D tmp = floor((abs(Ve) - sqrt(mi_me)*sound_speed),0.0);
      Ve_dampening = -(Ve/abs(Ve))*Ve_supsonic_factor * (exp(tmp)-1.0);
      ddt(VePsi) += Ve_dampening;
    }

    

  }

  ///////////////////////////////////////////////////////////
  // Ion velocity
  if (ion_velocity) {
    TRACE("Ion velocity");

    if (currents) {
      // ddt(NVi) = bracket(NVi, phi, BRACKET_ARAKAWA) * bracket_factor;
      // ExB drift, only if electric field calculated
      if (use_Div_n_bxGrad_f_B_XPPM){
	TE_NVi_ExB = -Div_n_bxGrad_f_B_XPPM(NVi, phi, ne_bndry_flux , poloidal_flows , false , bracket_factor) * scale_ExB;
      } else {
	TE_NVi_ExB = -bracket(phi,NVi, BRACKET_ARAKAWA) * bracket_factor * scale_ExB;
      }
      ddt(NVi) = TE_NVi_ExB;

      
    } else {
      ddt(NVi) = 0.0;
    }

    if (MMS_Ne_ParDiff> 0.0){
      auto tmp = Div_par_K_Grad_par(a_MMS3d,NVi);
      ddt(NVi) += tmp;
    }

    
    if (j_diamag) {
      // Magnetic drift
      TE_NVi_dia = -fci_curvature(mul_all(NVi , Ti),use_bracket);
      ddt(NVi) += TE_NVi_dia;
    }

    // FV with added dissipation
    if (MMS_Ne_ParDiff <= 0.0){
      if (use_Div_parP_n){
	TE_NVi_parflow = -Div_parP_n(Ne, Vi, sound_speed, fwd_bndry_mask, bwd_bndry_mask);
      } else {
	
	auto nvivi = mul_all(NVi,Vi);
	TE_NVi_parflow = -Div_par(nvivi);
      }
      ddt(NVi) += TE_NVi_parflow;

    }

    // Ignoring polarisation drift for now
    if (pe_par) {
      Field3D peppi = add_all(Pe, Pi);
      TE_NVi_pe_par = -Grad_parP(peppi);
      ddt(NVi) += TE_NVi_pe_par;
    }

    if(ion_viscosity_par){
      auto tmp = Div_par_K_Grad_par(div_all(mul_all(Pi,tau_i),coord->Bxy),mul_all(B12,Vi));
      TE_NVi_viscos = 1.28*B12*tmp;
      ddt(NVi) += TE_NVi_viscos;
    }

    // Parallel numerical diffusion
    
    if (bool_numdiff) {
      for(auto &i : NVi.getRegion("RGN_NOBNDRY")) {
        TE_NVi_numdiff[i] = numdiff[i]*(NVi.ydown()[i.ym()] - 2.*NVi[i] + NVi.yup()[i.yp()]);
      }
      ddt(NVi) += TE_NVi_numdiff;
    }
    /*
    if (classical_diffusion) {
      // Using same cross-field drift as in density equation
      Field3D ViDn = mul_all(Vi,Dn);
      ddt(NVi) += FCIDiv_a_Grad_perp(ViDn, Ne);
      Field3D NVi_tauB2 = div_all(NVi, tauemimeSQB);
      ddt(NVi) += FCIDiv_a_Grad_perp(NVi_tauB2, TiTediff);
    }
    */
    
    
    TE_NVi_anom = 0.0;
    if ((anomalous_D > 0.0) && anomalous_D_nvi) {
      TE_NVi_anom += FCIDiv_a_Grad_perp(mul_all(Vi, a_d3d), Ne);
    }

    if (anomalous_nu > 0.0) {
      TE_NVi_anom += FCIDiv_a_Grad_perp(mul_all(Ne, a_nu3d), Vi); 
    }

    if((anomalous_nu > 0.0) || ((anomalous_D > 0.0) && anomalous_D_nvi)){
      ddt(NVi) += TE_NVi_anom;
    }

    if (bool_NVi_hyper){
      TE_NVi_hyper = -NVi_hyper*((SQ(SQ(coord->dx)))*D4DX4(NVi) + (SQ(SQ(coord->dz)))*D4DZ4(NVi));
      ddt(NVi) += TE_NVi_hyper;
    }

    if(NVi_supsonic_dissipation){
      Field3D tmp = floor((abs(Vi) - sound_speed),0.0);
      NVi_dampening = -(Vi/abs(Vi))*NVi_supsonic_factor * (exp(tmp)-1.0);
      ddt(NVi) += NVi_dampening;
    }


  }

  ///////////////////////////////////////////////////////////
  // Pressure equation
  TRACE("Electron pressure");

  if (evolve_te) {

    if (currents) {
      if(fci_transform){
         
	    if (use_Div_n_bxGrad_f_B_XPPM){
	      TE_Pe_ExB = -Div_n_bxGrad_f_B_XPPM(Pe, phi, pe_bndry_flux, poloidal_flows, true , bracket_factor) * scale_ExB;
	      ddt(Pe) = TE_Pe_ExB;
	    } else {
	      TE_Pe_ExB = -bracket(phi,Pe, BRACKET_ARAKAWA) * bracket_factor * scale_ExB;
	      ddt(Pe) = TE_Pe_ExB;
	    }


	    
      }else{
	if (use_Div_n_bxGrad_f_B_XPPM){
	  ddt(Pe) = -Div_n_bxGrad_f_B_XPPM(Pe, phi, pe_bndry_flux, poloidal_flows, true , bracket_factor);
	} else {
	  ddt(Pe) = -bracket(phi,Pe, BRACKET_ARAKAWA) * bracket_factor;
	}
	

	    
      }
    } else {
      ddt(Pe) = 0.0;
    }

    if (parallel_flow_p_term) {
      // Parallel flow
      if (fci_transform){
        //check_all(Pe);
        //check_all(Ve);
        Field3D peve = mul_all(Pe,Ve);
        TE_Pe_parflow = -Div_parP(peve);
        ddt(Pe) += TE_Pe_parflow;
      } else {
        if (currents) {
          ddt(Pe) -= FV::Div_par(Pe, Ve, sqrt(mi_me) * sound_speed);
        } else {
          ddt(Pe) -= FV::Div_par(Pe, Ve, sound_speed);
        }
      }
    }

    if (j_diamag) { // Diamagnetic flow
      // Magnetic drift (curvature) divergence.
      TE_Pe_dia = (5. / 3) * fci_curvature(mul_all(Pe , Te),use_bracket);
      ddt(Pe) += TE_Pe_dia;

      // This term energetically balances diamagnetic term
      // in the vorticity equation
      // ddt(Pe) -= (2. / 3) * Pe * (Curlb_B * Grad(phi));
      TE_Pe_energ_balance = -(2. / 3) * Pe * fci_curvature(phi,use_bracket);
      ddt(Pe) += TE_Pe_energ_balance;
    }

    // Parallel heat conduction
    if (thermal_conduction) {
      if (fci_transform) {
        //check_all(kappa_epar);
        TE_Pe_cond = (2. / 3) * Div_par_K_Grad_par(kappa_epar, Te);
        ddt(Pe) += TE_Pe_cond;
      } else {
        ddt(Pe) += (2. / 3) * FV::Div_par_K_Grad_par(kappa_epar, Te);
      }
    }

    if (thermal_flux) {
      // Parallel heat convection
      if (fci_transform) {
        Field3D tejpar = mul_all(Te,Jpar);
	TE_Pe_thermal_flux = (2. / 3) * 0.71 * Div_parP(tejpar);
        ddt(Pe) += TE_Pe_thermal_flux;
      } else {
        ddt(Pe) += (2. / 3) * 0.71 * Div_par(Te * Jpar);
      }
    }

    if (currents && resistivity) {
      // Ohmic heating
      TE_Pe_ohmic = nu * Jpar * (Jpar - Jpar0) / Ne;
      ddt(Pe) += TE_Pe_ohmic;
    }

    if (bool_pe_hyper) {
      auto tmp = ( (SQ(SQ(coord->dz)))  * D4DZ4(Pe) + SQ(SQ(coord->dx))*D4DX4(Pe)  );
      TE_Pe_hyper = -pe_hyper * tmp;
      ddt(Pe) += TE_Pe_hyper;
    }

    ///////////////////////////////////
    // Heat transmission through sheath

    wall_power = 0.0; // Diagnostic output
    if (parallel_sheaths){
      sheath_dpe = 0.;

      for (const auto &bndry_par :
           mesh->getBoundariesPar(BoundaryParType::xout)) {
	for (const auto &pnt : *bndry_par) {
          int x = pnt.ind().x();
          int y = pnt.ind().y();
          int z = pnt.ind().z();
          // Temperature and density at the sheath entrance
          BoutReal tesheath =
              floor(0.5 * (Te(x, y, z) +
                           Te.ynext(bndry_par->dir)(x, y + bndry_par->dir, z)),
                    0.0);
          BoutReal nesheath =
              floor(0.5 * (Ne(x, y, z) +
                           Ne.ynext(bndry_par->dir)(x, y + bndry_par->dir, z)),
                    0.0);
          BoutReal vesheath =
	    0.5 * (Ve(x, y, z) +
                           Ve.ynext(bndry_par->dir)(x, y + bndry_par->dir, z));
          // BoutReal tisheath = floor(
          //                               0.5 * (Ti(x, y, z) +
          // Ti.ynext(bndry_par->dir)(x, y + bndry_par->dir, z)),
          // 0.0);

          // Sound speed (normalised units)
          // BoutReal Cs =bndry_par->dir* sqrt(tesheath + tisheath);

          // Heat flux
          BoutReal q = floor((sheath_gamma_e - 1.5) * tesheath * nesheath * vesheath *
			     bndry_par->dir,0.0);
          // Multiply by cell area to get power
          BoutReal flux = q * coord->J(x, y, z) / sqrt(coord->g_22(x, y, z));

          // Divide by volume of cell, and 2/3 to get pressure
          BoutReal power =
            flux
            / (coord->dy(x, y, z) * coord->J(x, y, z));
          // ddt(Pe)(x, y, z) -= (2. / 3) * power;
          sheath_dpe(x, y, z) -= (2. / 3) * power;
        }
      }
      sheath_dpe.name = "sheath physics";
      ddt(Pe) += sheath_dpe;
    }


    // Transfer and source terms
    if (thermal_force) {
      TE_Pe_thermal_force = -(2. / 3) * 0.71 * Jpar * Grad_parP(Te);
      
      ddt(Pe) += TE_Pe_thermal_force;
    }

    if (pe_par_p_term) {
      // This term balances energetically the pressure term
      // in Ohm's law
      TE_Pe_par_p_term = -(2. / 3) * Pe * Div_parP(Ve);
      ddt(Pe) += TE_Pe_par_p_term;
    }

    //////////////////////
    // Classical diffusion
    /*
    if (classical_diffusion) {

      // Combined resistive drift and cross-field heat diffusion
      // nu_rho2 = nu_ei * rho_e^2 in normalised units
      Field3D nu_rho2 = div_all(Te, mul_all(mul_all(tau_e, mi_me), B42));
      Field3D PePi = add_all(Pe, Pi);
      Field3D nu_rho2Ne = mul_all(nu_rho2, Ne);
      ddt(Pe) += (2. / 3) * (FCIDiv_a_Grad_perp(nu_rho2, PePi) +
                             (11. / 12) * FCIDiv_a_Grad_perp(nu_rho2Ne, Te));
    }
    */

    //////////////////////
    // Anomalous diffusion
    TE_Pe_anom = 0.0;
    if ((anomalous_D > 0.0) && anomalous_D_pepi) {
      TE_Pe_anom += FCIDiv_a_Grad_perp(mul_all(a_d3d, Te), Ne);
    }
    if (anomalous_chi > 0.0) {
      TE_Pe_anom += (2. / 3) * FCIDiv_a_Grad_perp(mul_all(a_chi3d, Ne), Te);
    }
    ddt(Pe) += TE_Pe_anom;

    // hyper diffusion
    if (bool_numdiff) {
      BOUT_FOR(i, Pe.getRegion("RGN_NOBNDRY")) {
	TE_Pe_numdiff[i] = numdiff[i]*(Pe.ydown()[i.ym()] - 2.*Pe[i] + Pe.yup()[i.yp()]);
      }
      ddt(Pe) += TE_Pe_numdiff;
    }
    if (verbose){
      for (const auto& ind : Pe.getRegion("RGN_NOBNDRY")) {
	Pe_yup[ind] = Pe.yup()[ind.yp()];
	Pe_ydown[ind] = Pe.ydown()[ind.ym()];
	kappa_epar_yup[ind] = kappa_epar.yup()[ind.yp()];
        kappa_epar_ydown[ind] = kappa_epar.ydown()[ind.ym()];
      }

    }
    //////////////////////
    // Sources

    if (adapt_source) {
      // Add source. Ensure that sink will go to zero as Pe -> 0
      Field3D PeErr = averageY(DC(Pe) - PeTarget);

      if (core_sources) {
        // Sources only in core

        ddt(Spe) = 0.0;
        for (int x = mesh->xstart; x <= mesh->xend; x++) {
          if (!mesh->periodicY(x))
            continue; // Not periodic, so skip

          for (int y = mesh->ystart; y <= mesh->yend; y++) {
                for (int z = 0; z <= mesh->LocalNz; z++) {
                  Spe(x, y, z) -= source_p * PeErr(x, y, z);
                  ddt(Spe)(x, y, z) = -source_i * PeErr(x, y, z);

                  if (Spe(x, y, z) < 0.0) {
                    Spe(x, y, z) = 0.0;
                    if (ddt(Spe)(x, y, z) < 0.0)
                      ddt(Spe)(x, y, z) = 0.0;
                  }
            }
          }
        }

        if (energy_source) {
          // Add the same amount of energy to each particle
          PeSource = Spe * Ne / DC(Ne);
        } else {
          PeSource = Spe;
        }
      } else {

        Spe -= source_p * PeErr / PeTarget;
        ddt(Spe) = -source_i * PeErr;

        if (energy_source) {
          // Add the same amount of energy to each particle
          PeSource = Spe * Ne / DC(Ne);
        } else {
          PeSource = Spe * where(Spe, PeTarget, Pe);
        }
      }

      if (source_vary_g11) {
        PeSource *= g11norm;
      }

    } else {
      // Not adapting sources

      if (energy_source) {
        // Add the same amount of energy to each particle
        PeSource = Spe * Ne / DC(Ne);

        if (source_vary_g11) {
          PeSource *= g11norm;
        }
      } else {
        // Add the same amount of energy per volume
        // If no particle source added, then this can lead to
        // a small number of particles with a lot of energy!
      }
    }

    ddt(Pe) += PeSource;
  } else {
    ddt(Pe) = 0.0;
  }

  ///////////////////////////////////////////////////////////
  // Ion pressure equation
  // Similar to electron pressure equation
  TRACE("Ion pressure");

  if (evolve_ti) {

    if (currents) {
      if(fci_transform){
           
	    if (use_Div_n_bxGrad_f_B_XPPM){
	      ddt(Pi) = -Div_n_bxGrad_f_B_XPPM(Pi, phi, pe_bndry_flux, poloidal_flows, true , bracket_factor) * scale_ExB;
	    } else {
	      ddt(Pi) = -bracket(phi,Pi, BRACKET_ARAKAWA) * bracket_factor*scale_ExB;
	    } 
	    
      }else{
            // Divergence of heat flux due to ExB advection
	ddt(Pi) = -Div_n_bxGrad_f_B_XPPM(Pi, phi, pe_bndry_flux, poloidal_flows, true, bracket_factor);
      }
    } else {
      ddt(Pi) = 0.0;
    }

    // Parallel flow
    if (parallel_flow_p_term) {
      if (fci_transform) {
        //check_all(Pi);
        //check_all(Vi);
        Field3D pivi = mul_all(Pi,Vi);
        ddt(Pi) -= Div_parP(pivi);
      } else {
        ddt(Pi) -= FV::Div_par(Pi, Vi, sound_speed);
      }
    }

    if (j_diamag) { // Diamagnetic flow
      // Magnetic drift (curvature) divergence
      ddt(Pi) -= (5. / 3) * fci_curvature(mul_all(Pi , Ti),use_bracket);


      // Compression of ExB flow
      // These terms energetically balances diamagnetic term
      // in the vorticity equation
      // ddt(Pi) -= (2. / 3) * Pi * (Curlb_B * Grad(phi));
      ddt(Pi) -= (2. / 3) * Pi * fci_curvature(phi,use_bracket);

      if (fci_transform) {
        ddt(Pi) += Pi * fci_curvature(Pi + Pe,use_bracket);
      } else {
        ddt(Pi) += Pi * Div((Pe + Pi) * Curlb_B);
      }
    }

    if (j_par) {
      if (boussinesq) {
        ddt(Pi) -= (2. / 3) * Jpar * Grad_parP(Pi);
      } else {
        ddt(Pi) -= (2. / 3) * Jpar * Grad_parP(Pi) / Ne;
      }
    }

    // Parallel heat conduction
    if (thermal_conduction) {
      if (fci_transform) {
        ddt(Pi) += (2. / 3) * Div_par_K_Grad_par(kappa_ipar, Ti);
      } else {
        ddt(Pi) += (2. / 3) * FV::Div_par_K_Grad_par(kappa_ipar, Ti);
      }
    }

    // Parallel pressure gradients (sound waves)
    if (pe_par_p_term) {
      // This term balances energetically the pressure term
      // in the parallel momentum equation
      ddt(Pi) -= (2. / 3) * Pi * Div_parP(Vi);
    }

    if (electron_ion_transfer) {
      // Electron-ion heat transfer
      Wi = (3. / mi_me) * Ne * (Te - Ti) / tau_e;
      ddt(Pi) += (2. / 3) * Wi;
      ddt(Pe) -= (2. / 3) * Wi;
    }


    if (bool_pi_hyper) {
      auto tmp = -pi_hyper * ( (SQ(SQ(coord->dz)))  * D4DZ4(Pi) + SQ(SQ(coord->dx))*D4DX4(Pi)  );
      ddt(Pi) += tmp;
    }

    //////////////////////
    // Classical diffusion
    /*
    if (classical_diffusion) {
      Field3D Pi_B2tau, PePi, nu_rho2, nu_rho2Ne;
      alloc_all(Pi_B2tau);
      alloc_all(PePi);
      alloc_all(nu_rho2);
      alloc_all(nu_rho2Ne);
      BOUT_FOR(i, Pi.getRegion("RGN_ALL")) {
        // Cross-field heat conduction
        // kappa_perp = 2 * n * nu_ii * rho_i^2
        Pi_B2tau[i] = (2. * Pi[i]) / (B42[i] * tau_i[i]);
        nu_rho2[i] = Te[i] / (tau_e[i] * mi_me * B42[i]);
        PePi[i] = Pe[i] + Pi[i];
        nu_rho2Ne[i] = nu_rho2[i] * Ne[i];

        Pi_B2tau.yup()[i] =
            (2. * Pi.yup()[i]) / (B42.yup()[i] * tau_i.yup()[i]);
        nu_rho2.yup()[i] =
            Te.yup()[i] / (tau_e.yup()[i] * mi_me * B42.yup()[i]);
        PePi.yup()[i] = Pe.yup()[i] + Pi.yup()[i];
        nu_rho2Ne.yup()[i] = nu_rho2.yup()[i] * Ne.yup()[i];

        Pi_B2tau.ydown()[i] =
            (2. * Pi.ydown()[i]) / (B42.ydown()[i] * tau_i.ydown()[i]);
        nu_rho2.ydown()[i] =
            Te.ydown()[i] / (tau_e.ydown()[i] * mi_me * B42.ydown()[i]);
        PePi.ydown()[i] = Pe.ydown()[i] + Pi.ydown()[i];
        nu_rho2Ne.ydown()[i] = nu_rho2.ydown()[i] * Ne.ydown()[i];
      }

      // BOUT_FOR(i, Pi.getRegion("RGN_NOBNDRY")) {
      ddt(Pi) += (2. / 3) * FCIDiv_a_Grad_perp(Pi_B2tau, Ti);

      // Resistive drift terms

      // mesh->communicate(nu_rho2Ne,Te);
      ddt(Pi) += (5. / 3) * (FCIDiv_a_Grad_perp(nu_rho2, PePi) -
                             (1.5) * FCIDiv_a_Grad_perp(nu_rho2Ne, Te));

      // Collisional heating from perpendicular viscosity
      // in the vorticity equation

      if (currents) {
        Vector3D Grad_perp_vort = Grad(Vort);
        Field3D phiPi = add_all(phi, Pi);
        Grad_perp_vort.y = 0.0; // Zero parallel component
        ddt(Pi) -= (2. / 3) * (3. / 10) * Ti / (SQ(coord->Bxy) * tau_i)
                   * (Grad_perp_vort * Grad(phiPi));
      }
    }
    */

    //////////////////////
    // Anomalous diffusion

    if ((anomalous_D > 0.0) && anomalous_D_pepi) {
      ddt(Pi) += FCIDiv_a_Grad_perp(mul_all(a_d3d, Ti), Ne);
    }

    if (anomalous_chi > 0.0) {
      ddt(Pi) += (2. / 3) * FCIDiv_a_Grad_perp(mul_all(a_chi3d, Ne), Ti);
    }

    // hyper diffusion
    if (bool_numdiff) {
      BOUT_FOR(i, Pi.getRegion("RGN_NOBNDRY")) {
        ddt(Pi)[i] += numdiff[i]*(Pi.ydown()[i.ym()] - 2.*Pi[i] + Pi.yup()[i.yp()]);
      }
    }

    ///////////////////////////////////
    // Heat transmission through sheath

    if (parallel_sheaths){
      sheath_dpi = 0.0;
      for (const auto &bndry_par :
           mesh->getBoundariesPar(BoundaryParType::xout)) {
	for (const auto &pnt : *bndry_par) {
          int x = pnt.ind().x();
          int y = pnt.ind().y();
          int z = pnt.ind().z();
          // Temperature and density at the sheath entrance
          BoutReal tisheath =
              floor(0.5 * (Ti(x, y, z) +
                           Ti.ynext(bndry_par->dir)(x, y + bndry_par->dir, z)),
                    0.0);
          BoutReal tesheath =
              floor(0.5 * (Te(x, y, z) +
                           Te.ynext(bndry_par->dir)(x, y + bndry_par->dir, z)),
                    0.0);
          BoutReal nesheath =
              floor(0.5 * (Ne(x, y, z) +
                           Ne.ynext(bndry_par->dir)(x, y + bndry_par->dir, z)),
                    0.0);
          BoutReal visheath =
              0.5 * (Vi(x, y, z) +
                     Vi.ynext(bndry_par->dir)(x, y + bndry_par->dir, z));

          // Sound speed (normalisexd units)
          // BoutReal Cs = bndry_par->dir * sqrt(tesheath + tisheath);

          // Heat flux
          BoutReal q = (sheath_gamma_i - 1.5) * tisheath * nesheath * visheath *
                       bndry_par->dir;

          // Multiply by cell area to get power
          BoutReal flux = q * coord->J(x, y, z) / sqrt(coord->g_22(x, y, z));

          // Divide by volume of cell, and 2/3 to get pressure
          BoutReal power =
            flux
            / (coord->dy(x, y, z) * coord->J(x, y, z));
          sheath_dpi(x, y, z) -= (3. / 2) * power;
        }
      }
      ddt(Pi) += sheath_dpi;
    }

    //////////////////////
    // Sources

    if (adapt_source) {
      // Add source. Ensure that sink will go to zero as Pe -> 0
      Field3D PiErr = averageY(DC(Pi) - PiTarget);

      if (core_sources) {
        // Sources only in core

        ddt(Spi) = 0.0;
        for (int x = mesh->xstart; x <= mesh->xend; x++) {
          if (!mesh->periodicY(x))
            continue; // Not periodic, so skip

          for (int y = mesh->ystart; y <= mesh->yend; y++) {
                for (int z = 0; z <= mesh->LocalNz; z++) {
                  Spi(x, y, z) -= source_p * PiErr(x, y, z);
                  ddt(Spi)(x, y, z) = -source_i * PiErr(x, y, z);

                  if (Spi(x, y, z) < 0.0) {
                    Spi(x, y, z) = 0.0;
                    if (ddt(Spi)(x, y, z) < 0.0)
                      ddt(Spi)(x, y, z) = 0.0;
                  }
            }
          }
        }

        if (energy_source) {
          // Add the same amount of energy to each particle
          PiSource = Spi * Ne / DC(Ne);
        } else {
          PiSource = Spi;
        }
      } else {

        Spi -= source_p * PiErr / PiTarget;
        ddt(Spi) = -source_i * PiErr;

        if (energy_source) {
          // Add the same amount of energy to each particle
          PiSource = Spi * Ne / DC(Ne);
        } else {
          PiSource = Spi * where(Spi, PiTarget, Pi);
        }
      }

      if (source_vary_g11) {
        PiSource *= g11norm;
      }

    } else {
      // Not adapting sources

      if (energy_source) {
        // Add the same amount of energy to each particle
        PiSource = Spi * Ne / DC(Ne);

        if (source_vary_g11) {
          PiSource *= g11norm;
        }

      } else {
        // Add the same amount of energy per volume
        // If no particle source added, then this can lead to
        // a small number of particles with a lot of energy!
      }
    }

    ddt(Pi) += PiSource;

  } else {
    ddt(Pi) = 0.0;
  }


  if (!evolve_plasma) {
    ddt(Ne) = 0.0;
    ddt(Pe) = 0.0;
    ddt(Vort) = 0.0;
    ddt(VePsi) = 0.0;
    ddt(NVi) = 0.0;
  }

  return 0;
} // rhs

/*!
 * Preconditioner. Solves the heat conduction
 *
 * @param[in] t  The simulation time
 * @param[in] gamma   Factor in front of the Jacobian in (I - gamma*J). Related
 * to timestep
 * @param[in] delta   Not used here
 */
int Hermes::precon(BoutReal t, BoutReal gamma, BoutReal delta) {
  static std::unique_ptr<InvertPar> inv{nullptr};
  if (!inv) {
    // Initialise parallel inversion class
    auto inv = InvertPar::create();
    inv->setCoefA(1.0);
  }
  if (thermal_conduction) {
    // Set the coefficient in front of Grad2_par2
    inv->setCoefB(-(2. / 3) * gamma * kappa_epar);
    Field3D dT = ddt(Pe);
    dT.applyBoundary("neumann");
    ddt(Pe) = inv->solve(dT);
  }

  // Neutral gas preconditioning
  if (neutrals)
    neutrals->precon(t, gamma, delta);

  return 0;
}

Field3D Hermes::fci_curvature(const Field3D &f, const bool &bool_bracket) {
  // Field3D result = mul_all(bracket(logB, f, BRACKET_ARAKAWA), bracket_factor);
  // mesh->communicate(result);
  if (bool_bracket){
    return 2 * bracket(logB, f, BRACKET_ARAKAWA) * bracket_factor;
  } else {
    //throw;
    // nabla (f nabla x (b/B)) = (dx,dy,dy)*(f*(bxcvx,bxcvy,bxcvz))
    //
    //                         = dx(f*bxcvx) + dz(f*bxcvz)
    //
    // !!!!!!! NOT SURE IF I NEED THE BRACKET_FACTOR

    return Div_f_v_no_y(f,bxcvx,bxcvz,false) * bracket_factor;
    //return FV::Div_f_v(f,bxcv,false) * bracket_factor;
  }
  
}





Field3D Hermes::Grad_parP(const Field3D &f) {
  return Grad_par(f); //+ 0.5*beta_e*bracket(psi, f, BRACKET_ARAKAWA);
}

Field3D Hermes::Div_parP(const Field3D &f) {
  return Div_par(f);
}

Field3D MinMod(const Field3D &f) {
  // get gradient in y direction, avoiding numerical issues
  Field3D result;
  result.allocate();
  BOUT_FOR(i, f.getRegion("RGN_NOBNDRY")) {
    const BoutReal fp = f.yup()[i.yp()];
    const BoutReal fm = f.ydown()[i.ym()];
    const BoutReal fi = f[i];
    const BoutReal gp = fp - fi;
    const BoutReal gm = fi - fm;
    if ((gp * gm) < 0) {
      result[i] = 0;
    } else if (abs(gp) < abs(gm)) {
      result[i] = gp;
    } else {
      result[i] = gm;
    }
    ASSERT2(std::isfinite(result[i]));
  }
  result.applyBoundary("neumann_o2");
  return result;
}

Field3D Hermes::Div_parP_f(const Field3D &f, const Field3D &v,
                           Field3D &sound_speed) {
  throw BoutException("NI");
}

Field3D Hermes::Div_parP_n(const Field3D &n, const Field3D &v,
                           Field3D &sound_speed, const BoutMask &fwd,
                           const BoutMask &bwd) {
  Field3D gn = MinMod(n);
  Field3D gv = MinMod(v);
  n.getMesh()->communicate(gn, gv, sound_speed);
  gn.applyParallelBoundary(parbc);
  gv.applyParallelBoundary(parbc);
  sound_speed.applyParallelBoundary(parbc);
  Field3D result{0.0};

  auto coord = n.getCoordinates();
  BOUT_FOR(i, n.getRegion("RGN_NOBNDRY")) {
    const auto ip = i.yp();
    const auto im = i.ym();

    // const BoutReal iVi =
    //     1 / (coord->dx[i] * coord->dy[i] * coord->dz[i] * coord->J[i]);
    // const BoutReal Ai =
    //     coord->dx[i] * coord->dz[i] * coord->J[i] / sqrt(coord->g_22[i]);
    // Area / Volume
    const BoutReal AoVi = 1 / (coord->dy[i] * sqrt(coord->g_22[i]));

    const BoutReal niR = n[i] + gn[i] / 2;
    const BoutReal viR = v[i] + gv[i] / 2;
    const BoutReal npL = n.yup()[ip] - gn.yup()[ip];
    const BoutReal vpL = v.yup()[ip] - gv.yup()[ip];
    const BoutReal niL = n[i] - gn[i] / 2;
    const BoutReal viL = v[i] - gv[i] / 2;
    const BoutReal nmR = n.ydown()[im] - gn.ydown()[im];
    const BoutReal vmR = v.ydown()[im] - gv.ydown()[im];
    //const BoutReal amaxp = std::max(
    //    {abs(v[i]), abs(v.yup()[ip]), sound_speed[i], sound_speed.yup()[ip]});
    //const BoutReal amaxm = std::max({abs(v[i]), abs(v.ydown()[im]),
    //                                sound_speed[i], sound_speed.ydown()[im]});
    const BoutReal amaxp = std::max({abs(v[i]), abs(v.yup()[ip]), sound_speed[i], sound_speed.yup()[ip]});
    const BoutReal amaxm = std::max({abs(v[i]), abs(v.ydown()[im]),sound_speed[i], sound_speed.ydown()[im]});

    
    BoutReal Gnvp = 0.5 * (niR * SQ(viR) + npL * SQ(vpL)) +
                    0.5 * amaxp * (niR * viR - npL * vpL);
    BoutReal Gnvm = 0.5 * (nmR * SQ(vmR) + niL * SQ(viL)) +
                    0.5 * amaxm * (nmR * vmR - niL * viL);
    if (Div_parP_n_sheath_extra) {
      if (fwd[i]) {
        const BoutReal vip = 0.5 * (v[i] + v.yup()[ip]);
        const BoutReal nip = 0.5 * (n[i] + n.yup()[ip]);
        Gnvp = niR * viR * vip + amaxp * (niR * viR - nip * vip);
      }
      if (bwd[i]) {
        const BoutReal vim = 0.5 * (v[i] + v.ydown()[im]);
        const BoutReal nim = 0.5 * (n[i] + n.ydown()[im]);
        Gnvp = niL * viL * vim + amaxm * (niL * viL - nim * vim);
      }
    }
    ASSERT1(std::isfinite(Gnvp));
    ASSERT1(std::isfinite(Gnvm));
    result[i] = AoVi * (Gnvp - Gnvm);
  }
  return result;
}

Field3D Hermes::FCIDiv_a_Grad_perp(const Field3D &a, const Field3D &f) {
  return (*_FCIDiv_a_Grad_perp)(a, f);
}

