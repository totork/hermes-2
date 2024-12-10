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

BoutReal limitFree(BoutReal fc, BoutReal fm, BoutReal floorval){
  if (fc>fm){
    return fc;
  }
  BoutReal fp = 2.0*fc - fm;
  return floor(fp,floorval);
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

  FiniteElMass = optsc["FiniteElMass"]
                        .doc("Use finite electron mass?")
                        .withDefault<bool>(true);

  calc_potential=optsc["calc_potential"]
                        .doc("Calculate the electrostatic potential?")
			.withDefault<bool>(true);
  
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
  Pi_anomalous = optpi["Pi_anomalous"].doc("Include anomalous effects in ion energy").withDefault<bool>(false);
  Pi_energyexchange = optpi["Pi_energyexchange"].doc("Include anomalous effects in ion energy").withDefault<bool>(false);
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

  if (optvort["bndry_xout"] == "dirichlet"){
    Vort_dirichlet=true;
  } else {
    Vort_dirichlet=false;
  }

  output.write("Vort_dirichlet {}\n", Vort_dirichlet);
  
  // bool VePsi_parefield, VePsi_parpressure, VePsi_partemp, VePsi_parcurrent, VePsi_ExB, VePsi_parflow;

  VePsi_parefield = optvepsi["VePsi_parefield"].doc("Use parallel electric field in electron velocity").withDefault<bool>(false);
  VePsi_parpressure = optvepsi["VePsi_parpressure"].doc("Use parallel pressure gradient in electron velocity").withDefault<bool>(false);
  VePsi_partemp = optvepsi["VePsi_partemp"].doc("Use parallel temperature gradient in electron velocity").withDefault<bool>(false);
  VePsi_parcurrent = optvepsi["VePsi_parcurrent"].doc("Use parallel current in electron velocity").withDefault<bool>(false);
  VePsi_ExB = optvepsi["VePsi_ExB"].doc("Use ExB drift in electron velocity").withDefault<bool>(false);
  VePsi_parflow = optvepsi["VePsi_parflow"].doc("Use parallel flow effects in electron velocity").withDefault<bool>(false);
  VePsi_hyper = optvepsi["VePsi_hyper"].doc("Use hyperdiffusion in electron velocity").withDefault<bool>(false);
  VePsi_numdiff = optvepsi["VePsi_numdiff"].doc("Use parallel numerical diffusion in electron velocity").withDefault<bool>(false);
  VePsi_parallelvisc = optvepsi["VePsi_parallelvisc"].doc("Use parallel viscosity as diffusion in electron velocity").withDefault<bool>(false);
  VePsi_supsonicdampening = optvepsi["VePsi_supsonicdampening"].doc("Use supersonic dampening in electron velocity").withDefault<bool>(false);
  VePsi_anomalous = optvepsi["VePsi_anomalous"].doc("Use anomalous transport in electron velocity").withDefault<bool>(false);
  
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
  TE_Pi_anomalous = 0.0;
  TE_Pi_energyexchange = 0.0;
  if (TE_Pi) {
    SAVE_REPEAT(TE_Pi_ExB, TE_Pi_mag, TE_Pi_parflow, TE_Pi_conduction, TE_Pi_diamagenergyexchange, TE_Pi_parviscousheat);
    SAVE_REPEAT(TE_Pi_resistivedrift, TE_Pi_perpviscous, TE_Pi_sources, TE_Pi_hyper, TE_Pi_numdiff,TE_Pi_anomalous,TE_Pi_energyexchange);
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
  TE_VePsi_parallelvisc = 0.0;
  TE_VePsi_supsonicdampening = 0.0;
  TE_VePsi_anomalous = 0.0;
  if (TE_VePsi) {
    SAVE_REPEAT(TE_VePsi_parefield, TE_VePsi_parpressure, TE_VePsi_partemp, TE_VePsi_parcurrent, TE_VePsi_ExB, TE_VePsi_parflow);
    SAVE_REPEAT(TE_VePsi_hyper, TE_VePsi_numdiff,TE_VePsi_parallelvisc,TE_VePsi_supsonicdampening, TE_VePsi_anomalous);
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

  OPTION(optnumerics, scale_ExB, 1.0);
  OPTION(optnumerics, resistivity_multiply, 1.0);
  OPTION(optnumerics, electron_weight, 1.0);
  OPTION(optnumerics, poloidal_flows, false);

  OPTION(optvepsi, Ve_supsonic_factor, 1.0);

  OPTION(optnumerics, floor_Ne,5e-2);
  OPTION(optnumerics, floor_Te,0.1);
  OPTION(optnumerics, floor_Ti,0.1);
  
  // Sheath switches
  
  OPTION(optsheath, sheath_model, 0);
  OPTION(optsheath, sheath_gamma_e, 7.0);
  OPTION(optsheath, sheath_gamma_i, 3.0);
  OPTION(optsheath, sheath_infsink, false);
  OPTION(optsheath, infsink_Te, 2.0);
  OPTION(optsheath, infsink_amp, 1.0);
  OPTION(optsheath, neutral_vwall, 1. / 3);  // 1/3rd Franck-Condon energy at wall
  OPTION(optsheath, sheath_yup, true);       // Apply sheath at yup?
  OPTION(optsheath, sheath_ydown, true);     // Apply sheath at ydown?
  OPTION(optsheath, test_boundaries, false); // Test boundary conditions
  OPTION(optsheath, parallel_sheaths, false); // Apply parallel sheath conditions?
  OPTION(optsheath, par_sheath_model, 0);
  OPTION(optsheath, par_sheath_ve, true);
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


  anomalous_D = opttransport["anomalous_D"].doc("Anomalous diffusion").withDefault(0.0);
  anomalous_nu = opttransport["anomalous_nu"].doc("Anomalous viscosity").withDefault(0.0);
  anomalous_chi = opttransport["anomalous_chi"].doc("Anomalous condoctivity").withDefault(0.0);

  hyper_D = opttransport["hyper_D"].doc("hyperdiffusion").withDefault(Field3D{0.0});
  hyper_chi = opttransport["hyper_chi"].doc("hyperconductivity").withDefault(Field3D{0.0});
  hyper_nu = opttransport["hyper_nu"].doc("hyperviscosity").withDefault(Field3D{0.0});

  num_D = opttransport["num_D"].doc("numerical parallel diffusion").withDefault(0.0);
  num_nu = opttransport["num_nu"].doc("numerical parallel viscosity").withDefault(0.0);
  num_chi = opttransport["num_chi"].doc("numerical parallel conductivity").withDefault(0.0);

  
  hyper_D /= (rho_s0 * rho_s0 * rho_s0 * rho_s0) * Omega_ci;
  hyper_nu /= (rho_s0 * rho_s0 * rho_s0 * rho_s0) * Omega_ci;
  hyper_chi /= (rho_s0 * rho_s0 * rho_s0 * rho_s0) * Omega_ci;

  num_D /= rho_s0 * rho_s0 * Omega_ci;
  num_nu /= rho_s0 * rho_s0 * Omega_ci;
  num_chi /= rho_s0 * rho_s0 * Omega_ci;

  
  hyper_D.applyBoundary("neumann");
  hyper_chi.applyBoundary("neumann");
  hyper_nu.applyBoundary("neumann");
  num_D.applyBoundary("neumann");
  num_chi.applyBoundary("neumann");
  num_nu.applyBoundary("neumann");

  mesh->communicate( hyper_D , hyper_chi , hyper_nu , num_D , num_nu ,num_chi);
  
  hyper_D.applyParallelBoundary("parallel_neumann_o1");
  hyper_nu.applyParallelBoundary("parallel_neumann_o1");
  hyper_chi.applyParallelBoundary("parallel_neumann_o1");
  num_D.applyParallelBoundary("parallel_neumann_o1");
  num_nu.applyParallelBoundary("parallel_neumann_o1");
  num_chi.applyParallelBoundary("parallel_neumann_o1");


  
  
  if (anomalous_D > 0.0) {
    // Normalise
    anomalous_D /= rho_s0 * rho_s0 * Omega_ci; // m^2/s
    output.write("\tnormalised anomalous D_perp = {:e}\n", anomalous_D);
    a_d3d = anomalous_D;
    a_d3d.applyBoundary("neumann");
    mesh->communicate(a_d3d);
    a_d3d.applyParallelBoundary("parallel_neumann_o1");
  }

  
  
  if (anomalous_chi > 0.0) {
    // Normalise
    anomalous_chi /= rho_s0 * rho_s0 * Omega_ci; // m^2/s
    output.write("\tnormalised anomalous chi_perp = {:e}\n", anomalous_chi);
    a_chi3d = anomalous_chi;
    a_chi3d.applyBoundary("neumann");
    mesh->communicate(a_chi3d);
    a_chi3d.applyParallelBoundary("parallel_neumann_o1");
    
  }
  if (anomalous_nu > 0.0) {
    // Normalise
    anomalous_nu /= rho_s0 * rho_s0 * Omega_ci; // m^2/s
    output.write("\tnormalised anomalous nu_perp = {:e}\n", anomalous_nu);
    a_nu3d = anomalous_nu;
    a_nu3d.applyBoundary("neumann");
    mesh->communicate(a_nu3d);
    a_nu3d.applyParallelBoundary("parallel_neumann_o1");
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


  SQSQ_g_11 = (SQ_all(coord->g_11));
  SQSQ_g_33 = (SQ_all(coord->g_33));

  
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
  /*
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
  */

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

  SAVE_REPEAT(phi,psi);
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
  Ve = 0.0;
  Vi = 0.0;
  Jpar = 0.0;

  
  
  phi.setBoundary("phi"); // For y boundaries                                                                                                     

  restart.addOnce(phi, "phi");
  
  aparSolver = LaplaceXZ::create(mesh,&opt["aparSolver"],CELL_CENTRE);
  
  Ve.setBoundary("Ve");
  nu.setBoundary("nu");
  Jpar.setBoundary("Jpar");


  SAVE_REPEAT(Ve,Vi,Jpar);
  psi = 0.0;
  nu = 0.0;
  kappa_epar = 0.0;
  kappa_ipar = 0.0;
  eta_epar = 0.0;
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


  Vi_sheath = 0.0;
  Ve_sheath = 0.0;
  Ne_sheath = 0.0;
  Te_sheath = 0.0;
  Ti_sheath = 0.0;
  Vort_sheath = 0.0;
  

  
  debug_Pe_conduction_A = 0.0;
  debug_Pe_conduction_B = 0.0;
  debug_sheath_infsink = 0.0;
  SAVE_REPEAT(Te, Ti);
  if (verbose) {
    // Save additional fields
    SAVE_REPEAT(debug_soundspeed,debug_phibndry3d);
    SAVE_REPEAT(tau_e, tau_i);
    SAVE_REPEAT(debug_Pe_conduction_A,debug_Pe_conduction_B);
    SAVE_REPEAT(Ne_sheath,Ve_sheath,Vi_sheath,Te_sheath,Ti_sheath,Vort_sheath);
    if(NVi_supsonic_dissipation){
      SAVE_REPEAT(NVi_dampening);
    }
    if(sheath_infsink){
      SAVE_REPEAT(debug_sheath_infsink);
    }
    if(Ve_supsonic_dissipation){
      SAVE_REPEAT(Ve_dampening);
    }
    
    if(kappa_limit_alpha>0.0){
      SAVE_REPEAT(debug_denom);
    }
    
    SAVE_REPEAT(kappa_epar,eta_epar); // Parallel electron heat conductivity
    SAVE_REPEAT(kappa_ipar); // Parallel ion heat conductivity
    SAVE_REPEAT(nu);
    SAVE_REPEAT(debug_visheath,debug_vesheath,debug_sheathexp);
    SAVE_REPEAT(debug_phisheath);
    SAVE_REPEAT(debug_VePsisheath);

  }

  zero_all(phi);
  zero_all(psi);



  
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


  // Here are some sanity checks for the flags

  if (evolve_vort && !calc_potential){
    throw BoutException("Evolving vorticity but not the potential");
  }


  
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


  Ne.applyBoundary();
  NVi.applyBoundary();
  Pe.applyBoundary();
  Vort.applyBoundary();
  Pi.applyBoundary();
  VePsi.applyBoundary();
  
  mesh->communicate(EvolvingVars);
  Ne.applyParallelBoundary(parbc);
  Vort.applyParallelBoundary(parbc);
  if (evolve_te){
    Pe.applyParallelBoundary(parbc);
  }
  if (evolve_ti){
    Pi.applyParallelBoundary(parbc);
  }
  NVi.applyParallelBoundary(parbc);
  
  if (evolve_vepsi){
    VePsi.applyParallelBoundary(parbc);
  }

  Field3D sound_speed;
  sound_speed.allocate();

  alloc_all(Te);
  alloc_all(Ti);
  alloc_all(Vi);
  alloc_all(Pi);
  alloc_all(Pe);

  
  
  BOUT_FOR(i, Ne.getRegion("RGN_ALL")) {
    floor_all(Ne, floor_Ne, i);

    if (!evolve_te) {
      copy_all(Pe, Ne, i); // Fixed electron temperature
    }

    div_all(Te, Pe, Ne, i);
    // ASSERT0(Te[i] > 1e-10);
    /// printf("%f\n", Te[i]);
    div_all(Vi, NVi, Ne, i);

    floor_all(Te, floor_Te, i);
    // ASSERT0(Te[i] > 1e-10);
    
    mul_all(Pe, Te, Ne, i);

    if (!evolve_ti) {
      copy_all(Pi, Ne, i); // Fixed ion temperature
    }

    div_all(Ti, Pi, Ne, i);
    floor_all(Ti, floor_Ti, i);
    mul_all(Pi, Ti, Ne, i);
    div_all(Te, Pe, Ne, i);
    // ASSERT0(Te[i] > 1e-10);

    sound_speed[i] =  sqrt(Te[i] + Ti[i] * (5. / 3));
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

  Field3D Ti32= pow(Ti,1.5);
  Ti32.applyBoundary("neumann");
  mesh->communicate(Ti32);
  Ti32.applyParallelBoundary(parbc);
  
  //////////////////////////////////////////////////////////////
  // Calculate electrostatic potential phi

  TRACE("Electrostatic potential");
  if (calc_potential){
    Field3D phi_boundary3d;
    phi_boundary3d = 0.0;
  

    if (boussinesq) {
    
      if (mesh->firstX()) {
	for (int j = mesh->ystart; j <= mesh->yend; j++) {
	  for (int k = 0; k < mesh->LocalNz; k++) {
	    phi_boundary3d(mesh->xstart - 1, j, k) = 0.5 * ( 3.0*(Te(mesh->xstart - 1, j, k) + Te(mesh->xstart, j, k)) + Pi(mesh->xstart - 1, j, k) + Pi(mesh->xstart, j, k));
	  }
	}
      }
    
    
      if (mesh->lastX()) {
	for (int j = mesh->ystart; j <= mesh->yend; j++) {
	  for (int k = 0; k < mesh->LocalNz; k++) {
	    phi_boundary3d(mesh->xend + 1, j, k) = 0.5 * ( 3.0*( Te(mesh->xend + 1, j, k) + Te(mesh->xend, j, k) ) + Pi(mesh->xend + 1, j, k) + Pi(mesh->xend, j, k) );
	    
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
      //phi.applyBoundary("neumann");
      mesh->communicate(phi);
      phi.applyParallelBoundary(parbc);
      
      phi = sub_all(phi, Pi);
      mesh->communicate(phi);
      phi.applyParallelBoundary(parbc);
    } else {
      ////////////////////////////////////////////
      // Non-Boussinesq
      //
      throw BoutException("Non-Boussinesq not implemented yet");
    }
    
  } else {
    phi = 0.0;
  } // End calc_potential
  


  

  //////////////////////////////////////////////////////////////
  // Calculate perturbed magnetic field psi
  TRACE("Calculating psi");

  
  if (electromagnetic) {
    if (FiniteElMass) {
      // Solve Helmholtz equation for psi
      auto tmp = -Ne*0.5*mi_me*beta_e;
      
      aparSolver->setCoefs(1.0,tmp);
      
      psi = aparSolver->solve(-Ne*VePsi,psi);
      mesh->communicate(psi);
      
      psi.applyParallelBoundary(parbc);
      
      Ve = VePsi - 0.5 * beta_e * mi_me * psi + Vi;
	
      Ve.applyBoundary("neumann");
      mesh->communicate(Ve);
      Ve.applyParallelBoundary(parbc);
      
    } else {
      throw BoutException("Running without finite electron mass is not possible anymore!");
    }
    
  } else {
    // Electrostatic
    zero_all(psi);
    // No psi contribution to VePsi
    Ve = add_all(VePsi , Vi);
  }

  
  Jpar = sub_all(NVi,mul_all(Ne,Ve));

  /*
  Jpar.applyBoundary("neumann");
  mesh->communicate(Jpar);
  Jpar.applyParallelBoundary(parbc);
  */

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

	  if (Vort_dirichlet){
	    Vort.ynext(bndry_par->dir)(x, y+bndry_par->dir, z) = 0.0;
	  } else {
	    Vort.ynext(bndry_par->dir)(x, y+bndry_par->dir, z) = Vort(x, y, z);
	  }
          // Here zero-gradient Te, heat flux applied later
          Te.ynext(bndry_par->dir)(x, y+bndry_par->dir, z) = Te(x, y, z);
          Ti.ynext(bndry_par->dir)(x, y+bndry_par->dir, z) = Ti(x, y, z);

          Pe.ynext(bndry_par->dir)(x, y+bndry_par->dir, z) = Pe(x, y, z);
          Pi.ynext(bndry_par->dir)(x, y+bndry_par->dir, z) = Pi(x, y, z);

          // Dirichlet conditions
	  /*
	  if (electromagnetic || FiniteElMass){
	    VePsi.ynext(bndry_par->dir)(x, y+bndry_par->dir, z) = VePsisheath;
	  }
	  */
	  
          Vi.ynext(bndry_par->dir)(x, y+bndry_par->dir, z) = visheath;//2. * visheath - Vi(x, y, z);
          if (par_sheath_ve){
            Ve.ynext(bndry_par->dir)(x, y+bndry_par->dir, z) = vesheath;//2. * vesheath - Ve(x, y, z);
          }
          Jpar.ynext(bndry_par->dir)(x, y+bndry_par->dir, z) = jsheath;
            // 2. * jsheath - Jpar(x, y, z);
          NVi.ynext(bndry_par->dir)(x, y+bndry_par->dir, z) = nesheath * visheath;//
            // 1. * nesheath * visheath;// - NVi(x, y, z);
        }
      }// End sheath loop
      /*
      // Set inner to neumann for all variables
      for (const auto &bndry_par :
           mesh->getBoundariesPar(BoundaryParType::xin)) {
        for (const auto &pnt : *bndry_par)  {
	  int x = pnt.ind().x();
          int y = pnt.ind().y();
          int z = pnt.ind().z();
	  
	  Ne.ynext(bndry_par->dir)(x, y+bndry_par->dir, z) = Ne(x,y,z);
	  NVi.ynext(bndry_par->dir)(x, y+bndry_par->dir, z) = NVi(x,y,z);
	  Pe.ynext(bndry_par->dir)(x, y+bndry_par->dir, z) = Pe(x,y,z);
	  Pi.ynext(bndry_par->dir)(x, y+bndry_par->dir, z) = Pi(x,y,z);
	  Vort.ynext(bndry_par->dir)(x, y+bndry_par->dir, z) = Vort(x,y,z);
	  VePsi.ynext(bndry_par->dir)(x, y+bndry_par->dir, z) = VePsi(x,y,z);
	}
      }// End set inner
      */
      
      break;
    }
    case 1: { // insulating boundary      break;
      for (const auto &bndry_par : mesh->getBoundariesPar(BoundaryParType::xout)) {
        for (const auto &pnt : *bndry_par)  {
	  TRACE("Setting new parallel sheaths with linear interpolation");
	  int x = pnt.ind().x();
          int y = pnt.ind().y();
          int z = pnt.ind().z();
	  BoutReal bdir = bndry_par->dir;

	  // limitFree(BoutReal fc, BoutReal fm, BoutReal floorval)
	  // Set the densities and temperatures

	  Ne_sheath(x,y,z) = limitFree(Ne(x,y,z),Ne.ynext(bdir)(x, y-bdir, z),floor_Ne);

	  Te_sheath(x,y,z) = limitFree(Te(x,y,z),Te.ynext(bdir)(x, y-bdir, z),floor_Te);

	  Ti_sheath(x,y,z) = limitFree(Ti(x,y,z),Ti.ynext(bdir)(x, y-bdir, z),floor_Ti);

	  

	  
	  Ve_sheath(x,y,z) = 2.0 * Ve(x,y,z) - Ve(x,y-bdir,z);
	  Vi_sheath(x,y,z) = 2.0 * Vi(x,y,z) - Vi(x,y-bdir,z);
	}
      }
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
  
  tau_e = div_all(mul_all(mul_all(div_all(Cs0 , rho_s0) , tau_e0) , Te32) , Ne);
  tau_i = div_all(mul_all(mul_all(div_all(Cs0 , rho_s0) , tau_i0) , Ti32) , Ne);
  
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
    
    Field3D gradTe = Grad_par(Te);
    
    gradTe.applyBoundary("neumann");
    mesh->communicate(gradTe);
    gradTe.applyParallelBoundary("parallel_neumann_o1");
    
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
  
  // Electron parallel viscosity

  eta_epar = mul_all(0.973, mul_all(mi_me,mul_all(tau_e,Te)));

  
  //////////////////////////////////////////////////////////////                                                                        
  TRACE("Calculating resistivity");

  nu = resistivity_multiply / (1.96 * tau_e * mi_me);
  nu.applyBoundary("neumann");
  mesh->communicate(nu);
  nu.applyParallelBoundary(parbc);

  Wi = (3. / mi_me) * Ne * (Te - Ti) / tau_e;



  // UP UNTIL NOW I NEED                                                                                                                                                                                          
  // Jpar                                                                                                                                                                                                         
  // Vi                                                                                                                                                                                                           
  // Ve                                                                                                                                                                                                           
  // Te                                                                                                                                                                                                           
  // Ti
  // kappa_epar
  // kappa_ipar
  // nu
  // W
  // tau_e
  // tau_i


  
  
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////                                                                                                                                                                
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////                                                                                                                                                                 
  //                                                Density equation                                          //                                                                                                                                                                 
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////                                                                                                                                                               
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////



  ddt(Ne) = 0.0;
  if (evolve_ne){
    TRACE("Density");

    
    if (Ne_ExB){// Row 1 Term 1
      TRACE("Density ExB");
      
      if (use_Div_n_bxGrad_f_B_XPPM){
	TE_Ne_ExB = -Div_n_bxGrad_f_B_XPPM(Ne, phi, ne_bndry_flux, poloidal_flows,true,bracket_factor) * scale_ExB;
      } else {
	TE_Ne_ExB = -bracket(phi,Ne, BRACKET_ARAKAWA) * bracket_factor*scale_ExB;
      }
      ddt(Ne) += TE_Ne_ExB;
    }  // End Ne_ExB

    
    if (Ne_mag){// Row 1 Term 2
      TRACE("Density mag");
      TE_Ne_mag = fci_curvature(Pe , use_bracket);
      ddt(Ne) += TE_Ne_mag;
    }  // End Ne_mag


    if (Ne_parflow){// Row 2 
      TRACE("Density parflow");
      Field3D neve = mul_all(Ne,Ve);
      TE_Ne_parflow = -Div_parP(neve);
      ddt(Ne) += TE_Ne_parflow;
    }  // End Ne_parflow

    
    if (Ne_collision){// Row 3
      TRACE("Density collisions");
      throw BoutException("Density collisions not implemented");
    }  // End Ne_collision

    
    if (Ne_anomalous){// Row 4 
      TRACE("Density anomalous");
      TE_Ne_anomalous = FCIDiv_a_Grad_perp(a_d3d, Ne);
      ddt(Ne) += TE_Ne_anomalous;
    }  // End Ne_anomalous

    
    if (Ne_sources){//Row 5 Term 2
      TRACE("Density sources");
      TE_Ne_sources=NeSource;
      ddt(Ne) += TE_Ne_sources;
    }  // End Ne_sources


    if (Ne_hyper){
      TRACE("Density hyperdiffusion");
      TE_Ne_hyper = hyperdissipation(hyper_D,Ne);
      ddt(Ne) += TE_Ne_hyper; 
    } // End Ne_hyper


    if (Ne_numdiff){
      TRACE("Density numerical parallel diffusion");
      TE_Ne_numdiff = numericaldissipation(num_D,Ne);
      ddt(Ne) += TE_Ne_numdiff;
    } // End Ne_numdiff

    
  } //End evolve_ne
  
  

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////                                                                                                                                                                 
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////                                                                                                                                                                 
  //                                               Vorticity equation                                         //                                                                                                                                                                 
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////                                                                                                                                                               
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////
  

  
  ddt(Vort) = 0.0;
  if (evolve_vort){
    TRACE("Vorticity");
    
    if(Vort_mag){// Row 1 
      TRACE("Vort_mag");
      TE_Vort_mag = fci_curvature(add_all(Pi , Pe),use_bracket);
      ddt(Vort) += TE_Vort_mag;
    } //End Vort_mag

    
    if(Vort_parcurrent){// Row 2
      TRACE("Vort_parcurrent");
      TE_Vort_parcurrent = Div_parP(Jpar);
      ddt(Vort) += TE_Vort_parcurrent;
    } //End Vort_parcurrent

    
    if (Vort_polarcurrent){

      TRACE("Vort_polarcurrent");

      if(boussinesq){

	if (j_pol_pi){

	  throw BoutException("j_pol_pi not implemented!");

	}else if (j_pol_simplified) {// Row 3 Term 2
	  // use simplified polarization term from i.e. GBS                                                                                             
	  if (use_Div_n_bxGrad_f_B_XPPM){

	    TE_Vort_polarcurrent = -Div_n_bxGrad_f_B_XPPM(Vort, phi, vort_bndry_flux,
					     poloidal_flows, false , bracket_factor) * scale_ExB;
	    
	  } else {

	    TE_Vort_polarcurrent = -bracket(phi,Vort, BRACKET_ARAKAWA) * bracket_factor * scale_ExB;
	    
	  }

	  ddt(Vort) += TE_Vort_polarcurrent;

	} //End j_pol_pi

      } else {

	throw BoutException("Non-boussinesq not implemented");

      }  //End boussinesq

    } //End Vort_polarcurrent

    
    if (Vort_anomalous){//Row 6 
      TE_Vort_anomalous = FCIDiv_a_Grad_perp(a_nu3d, Vort);
      ddt(Vort) += TE_Vort_anomalous;
    } // End Vort_anomalous


    if (Vort_hyper){
      TRACE("Vorticity hyperdiffusion");
      TE_Vort_hyper = hyperdissipation(hyper_nu,Vort);
      ddt(Vort) += TE_Vort_hyper;
    } // End Vort_hyper


    if (Vort_numdiff){
      TRACE("Vorticity numerical parallel diffusion");
      TE_Vort_numdiff = numericaldissipation(num_nu,Vort);
      ddt(Vort) += TE_Vort_numdiff;
    } // End Vort_numdiff

    
  }  //End evolve_vort



  //////////////////////////////////////////////////////////////////////////////////////////////////////////////                                                                                                                                                                 
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////                                                                                                                                                                 
  //                                               Ohm's law equation                                         //                                                                                                                                                                 
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////                                                                                                                                                                 
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////

  
  
  ddt(VePsi) = 0.0;
  if (evolve_vepsi){
    TRACE("Ohm's law");
    
    if (VePsi_parefield){//Row 1 Term 1
      TE_VePsi_parefield = mi_me * Grad_par(phi);
      ddt(VePsi) += TE_VePsi_parefield;
    } //End VePsi_parefield

    
    if (VePsi_parpressure){//Row 1 Term 2
      TE_VePsi_parpressure = -mi_me * Grad_par(Pe) / Ne;
      ddt(VePsi) += TE_VePsi_parpressure;
    } //End VePsi_parpressure


    if (VePsi_partemp){//Row 1 Term 3
      TE_VePsi_partemp = -mi_me * 0.71 * Grad_par(Te);
      ddt(VePsi) += TE_VePsi_partemp;
    } //End VePsi_partemp

    
    if (VePsi_parcurrent){//Row 2
      TE_VePsi_parcurrent = mi_me * nu * (Vi - Ve);
      ddt(VePsi) += TE_VePsi_parcurrent;
    } //End VePsi_parcurrent


    if (VePsi_ExB){//Row 3 Term 1
      if(use_Div_n_bxGrad_f_B_XPPM){
	TE_VePsi_ExB = -Div_n_bxGrad_f_B_XPPM(Ve-Vi, phi, false,poloidal_flows , false, bracket_factor) * scale_ExB;
      } else {
	TE_VePsi_ExB = -bracket(phi , sub_all(Ve,Vi) , BRACKET_ARAKAWA) * bracket_factor * scale_ExB;
      }
      ddt(VePsi) += TE_VePsi_ExB;
    } // End VePsi_ExB

    
    if (VePsi_parflow){//Row 3 Term 2
      TE_VePsi_parflow = -Ve * Div_parP(sub_all(Ve,Vi));
      ddt(VePsi) += TE_VePsi_parflow;
    } // End VePsi_parflow

    
    /*
    if(Ve_supsonic_dissipation){
      Field3D tmp = floor((abs(Ve) - sqrt(mi_me)*sound_speed),0.0);
      Ve_dampening = -(Ve/abs(Ve))*Ve_supsonic_factor * (exp(tmp)-1.0);
      ddt(VePsi) += Ve_dampening;
    }
    */


    if (VePsi_hyper){
      TRACE("VePsi hyperdiffusion");
      TE_VePsi_hyper = hyperdissipation(hyper_nu,VePsi);
      ddt(VePsi) += TE_VePsi_hyper;
    } // End VePsi_hyper


    if (VePsi_numdiff){
      TRACE("VePsi numerical parallel diffusion");
      TE_VePsi_numdiff = numericaldissipation(num_nu,VePsi);
      ddt(VePsi) += TE_VePsi_numdiff;
    } // End VePsi_numdiff


    if (VePsi_parallelvisc){
      TRACE("VePsi parallel viscosity");
      //mesh->communicate(eta_epar);
      //eta_epar.applyParallelBoundary(parbc);
      /*
      Field3D gradVe = Grad_par(Ve);
      mesh->communicate(gradVe);
      gradVe.applyParallelBoundary(parbc);
      TE_VePsi_parallelvisc = Div_par(eta_epar)*gradVe + eta_epar * Div_par(gradVe);
      */
      TE_VePsi_parallelvisc = Div_par_K_Grad_par(eta_epar,Ve);
      ddt(VePsi) += TE_VePsi_parallelvisc; 
    } // End VePsi_parallelvisc


    if (VePsi_supsonicdampening){
      Field3D tmp = floor((abs(Ve) - sqrt(mi_me)*sound_speed),0.0);                                                                                        TE_VePsi_supsonicdampening = -(Ve/abs(Ve))*Ve_supsonic_factor * (exp(tmp)-1.0);                                                                        
      ddt(VePsi) += TE_VePsi_supsonicdampening;      
    } // End VePsi_supsonicdampening


    if (VePsi_anomalous){
      TRACE("VePsi anomalous");
      TE_VePsi_anomalous = FCIDiv_a_Grad_perp(a_nu3d, Ve);
    } // End VePsi_anomalous

    
  } //End evolve_vepsi

  

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////                                                                                                                                                                 
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////                                                                                                                                                                 
  //                                         Ion momentum equation                                            //                                                                                                                                                                 
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////                                                                                                                                                                 
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////

  
  
  ddt(NVi) = 0.0;
  TRACE("Ion momentum");
  if (evolve_nvi){

    
    if (NVi_ExB){//Row 1 Term 1
      if (use_Div_n_bxGrad_f_B_XPPM){
        TE_NVi_ExB = -Div_n_bxGrad_f_B_XPPM(NVi, phi, ne_bndry_flux , poloidal_flows , false , bracket_factor) * scale_ExB;
      } else {
        TE_NVi_ExB = -bracket(phi,NVi, BRACKET_ARAKAWA) * bracket_factor * scale_ExB;
      }
      ddt(NVi) += TE_NVi_ExB;
    } // End NVi_ExB


    if (NVi_mag){//Row 1 Term 3
      TE_NVi_mag = -fci_curvature(mul_all(NVi , Ti),use_bracket);
      ddt(NVi) += TE_NVi_mag;
    } // End NVi_mag


    if (NVi_parflow){//Row 1 Term 2
      auto nvivi = mul_all(NVi,Vi);
      TE_NVi_parflow = -Div_parP(nvivi);
      ddt(NVi) += TE_NVi_parflow;
    } // End NVi_parflow

    
    if (NVi_parpressure){//Row 2
      Field3D peppi = add_all(Pe, Pi);
      TE_NVi_parpressure = -Grad_parP(peppi);
      ddt(NVi) += TE_NVi_parpressure;
    } // End NVi_parpressure


    if (NVi_parviscos){//Row 3
      auto tmp = Div_par_K_Grad_par(div_all(mul_all(Pi,tau_i),coord->Bxy),mul_all(B12,Vi));
      TE_NVi_parviscos = 1.28*B12*tmp;
      ddt(NVi) += TE_NVi_parviscos;
    } // End NVi_parviscos


    if (NVi_collision){
      throw BoutException("NVi collisions not implemented!");
    }

    
    if (NVi_anomalous){//Row 5
      TE_NVi_anomalous = FCIDiv_a_Grad_perp(mul_all(Vi, a_d3d), Ne);
      TE_NVi_anomalous += FCIDiv_a_Grad_perp(mul_all(Ne, a_nu3d), Vi);
      ddt(NVi) += TE_NVi_anomalous;
    }


    if (NVi_hyper){
      TRACE("Ion momentum hyperdiffusion");
      TE_NVi_hyper = hyperdissipation(hyper_nu,NVi);
      ddt(NVi) += TE_NVi_hyper;
    } // End NVi_hyper


    if (NVi_numdiff){
      TRACE("Ion momentum numerical parallel diffusion");
      TE_NVi_numdiff = numericaldissipation(num_nu,NVi);
      ddt(NVi) += TE_NVi_numdiff;
    } // End NVi_numdiff

    
  } // End evolve_nvi



  //////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //                                       Electron pressure equation                                         //
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////

  

  ddt(Pe) = 0.0;
  if (evolve_te){
    TRACE("Electron pressure");

    
    if (Pe_ExB){//Row 1 Term 1
      TRACE("Pe_ExB");
      if (use_Div_n_bxGrad_f_B_XPPM){
	TE_Pe_ExB = -Div_n_bxGrad_f_B_XPPM(Pe, phi, pe_bndry_flux, poloidal_flows, true , bracket_factor) * scale_ExB;
      } else {
	TE_Pe_ExB = -bracket(phi,Pe, BRACKET_ARAKAWA) * bracket_factor * scale_ExB;
      }
      ddt(Pe) += TE_Pe_ExB;
    } // End Pe_ExB


    if (Pe_mag){//Row 1 Term 1 and Term 3
      TRACE("Pe_mag");
      TE_Pe_mag = (5. / 3) * fci_curvature(mul_all(Pe , Te),use_bracket);
      TE_Pe_mag += -(2. / 3) * Pe * fci_curvature(phi,use_bracket);
      ddt(Pe) += TE_Pe_mag;
    } // End Pe_mag


    if (Pe_parflow){//Row 2 
      // Parallel flow plus compression
      TRACE("Pe_parflow + compression");
      Field3D peve = mul_all(Pe,Ve);
      TE_Pe_parflow = -Div_parP(peve) - (2. / 3) * Pe * Div_parP(Ve);;
      ddt(Pe) += TE_Pe_parflow;
    } // End Pe_parflow


    if (Pe_conduction){//Row 3
      TRACE("Pe_conduction");
      /*
      Field3D gradTe = Grad_par(Te);
      mesh->communicate(gradTe);
      gradTe.applyParallelBoundary(parbc);
      TE_Pe_conduction = (2.0/3.0) * ( Div_par(kappa_epar)*gradTe + kappa_epar*Div_par(gradTe) );
      */

      if (verbose){
	debug_Pe_conduction_A = Div_par(kappa_epar) * Grad_par(Te);
	debug_Pe_conduction_B = kappa_epar * Grad2_par2(Te);
      }
      
      TE_Pe_conduction = (2. / 3) * Div_par_K_Grad_par(kappa_epar, Te);
      ddt(Pe) += TE_Pe_conduction;
    } // End Pe_conduction


    if (Pe_ohmic){//Row 4 Term 3
      TRACE("Pe_ohmic");
      TE_Pe_ohmic = nu * Jpar * (Jpar) / Ne;
      ddt(Pe) += TE_Pe_ohmic;
    } // End Pe_ohmic


    if (Pe_thermalforce){//Row 4 Term 2
      TE_Pe_thermalforce = -(2. / 3) * 0.71 * Jpar * Grad_parP(Te);
      ddt(Pe) += TE_Pe_thermalforce;
    } // End Pe_thermalforce


    if (Pe_thermalcurrent){//Row 4 Term 1
      Field3D tejpar = mul_all(Te,Jpar);
      TE_Pe_thermalcurrent = (2. / 3) * 0.71 * Div_parP(tejpar);
      ddt(Pe) += TE_Pe_thermalcurrent;
    } //End Pe_thermalcurrent


    if (Pe_collision){
      throw BoutException("Pe_collision not implemented!");
    } //End Pe_collision


    if (Pe_anomalous){//Row 6
      TRACE("Pe anomalous transport");
      TE_Pe_anomalous = FCIDiv_a_Grad_perp(mul_all(a_d3d, Te), Ne) + (2. / 3) * FCIDiv_a_Grad_perp(mul_all(a_chi3d, Ne), Te);
      ddt(Pe) += TE_Pe_anomalous;
    } // End Pe_anomalous


    if (Pe_energyexchange){//Row 7 Term 3
      TRACE("Pe energy exchange");
      TE_Pe_energyexchange = -(2. / 3) * Wi;
      ddt(Pe) += TE_Pe_energyexchange;
    } // End Pe_energyexchange


    if (Pe_sources){//Row 7 Term 1
      TRACE("Pe sources");
      TE_Pe_sources = PeSource;
      ddt(Pe) += TE_Pe_sources;
    } //End Pe_sources


    if (parallel_sheaths){
      TRACE("Parallel sheaths in electron pressure");
      wall_power = 0.0; // Diagnostic output
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

	  if(sheath_infsink){
	    // sheath_infsink , infsink_Te , infsink_amp ,  debug_sheath_infsink
	    BoutReal tmp = floor(Te(x,y,z)-infsink_Te, 0.0);
	    debug_sheath_infsink(x,y,z) = -infsink_amp * (exp(tmp) - 1.0);
	  }
        }
      }
      sheath_dpe.name = "sheath physics";
      ddt(Pe) += sheath_dpe;
      if (sheath_infsink){
	ddt(Pe) += debug_sheath_infsink;
      }
    } //End parallel_sheaths


    if (Pe_hyper){
      TRACE("Electron pressure hyperdiffusion");
      TE_Pe_hyper = hyperdissipation(hyper_chi,Pe);
      ddt(Pe) += TE_Pe_hyper;
    } // End Pe_hyper


    if (Pe_numdiff){
      TRACE("Electron pressure numerical parallel diffusion");
      TE_Pe_numdiff = numericaldissipation(num_chi,Pe);
      ddt(Pe) += TE_Pe_numdiff;
    } // End Pe_numdiff
    
    
  } // End evolve_te



  //////////////////////////////////////////////////////////////////////////////////////////////////////////////                                                                                                                                                                 
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////                                                                                                                                                                 
  //                                            Ion pressure equation                                         //                                                                                                                                                                 
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////                                                                                                                                                                 
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////

  //bool Pi_ExB, Pi_mag, Pi_parflow, Pi_conduction, Pi_diamagenergyexchange, Pi_parviscousheat;
  //bool Pi_resistivedrift, Pi_perpviscous, Pi_sources,Pi_hyper,Pi_numdiff,Pi_anomalous;

  ddt(Pi) = 0.0;
  if (evolve_ti){
    TRACE("Ion pressure");


    if (Pi_ExB){//Row 1 Term 1 and Term 3
      TRACE("Pi ExB");
      if (use_Div_n_bxGrad_f_B_XPPM){
	TE_Pi_ExB = -Div_n_bxGrad_f_B_XPPM(Pi, phi, pe_bndry_flux, poloidal_flows, true , bracket_factor) * scale_ExB;
      } else {
        TE_Pi_ExB = -bracket(phi,Pi, BRACKET_ARAKAWA) * bracket_factor*scale_ExB;
      }
      TE_Pi_ExB += -(2. / 3) * Pi * fci_curvature(phi,use_bracket);                      // Compression
      ddt(Pi) += TE_Pi_ExB;
    } //End Pi_ExB


    if (Pi_mag){//Row 1 Term 2
      TRACE("Pi magnetic drift");
      TE_Pi_mag = -(5. / 3) * fci_curvature(mul_all(Pi , Ti),use_bracket);         // Actual diamag drift, 1st row in manual
      ddt(Pi) += TE_Pi_mag;
    } //End Pi_mag


    if (Pi_parflow){//Row 2 Term 1 and Term 2
      TRACE("Pi parflow");
      Field3D pivi = mul_all(Pi,Vi);
      TE_Pi_parflow = -Div_parP(pivi);
      TE_Pi_parflow += -(2. / 3) * Pi * Div_parP(Vi);
      ddt(Pi) += TE_Pi_parflow;
    } // End Pi_parflow


    if (Pi_diamagenergyexchange){//Row 3 Term 1 and Term 2
      TRACE("Pi energy exchange with diamag flows");
      TE_Pi_diamagenergyexchange = -(2. / 3) * Jpar * Grad_parP(Pi);
      TE_Pi_diamagenergyexchange += Pi * fci_curvature(add_all(Pi , Pe),use_bracket);
      ddt(Pi) += TE_Pi_diamagenergyexchange;
    } // End Pi_diamagenergyexchange
    

    if (Pi_conduction){//Row 5 Term 1
      TRACE("Pi thermal conduction");

      /*
      Field3D gradTi = Grad_par(Ti);
      mesh->communicate(gradTi);
      gradTi.applyParallelBoundary(parbc);
      TE_Pi_conduction = (2.0/3.0) * ( Div_par(kappa_ipar)*gradTi + kappa_ipar*Div_par(gradTi) );
      */
      TE_Pi_conduction = (2. / 3) * Div_par_K_Grad_par(kappa_ipar, Ti);
      ddt(Pi) = TE_Pi_conduction;
    } // End Pi_conduction 

    
    if (Pi_resistivedrift){
      throw BoutException("Ion resistive drift not implemented!");
    } // End Pi_resistivedrift
    

    if (Pi_perpviscous){
      throw BoutException("Ion pressure perpendicular viscosity heating not implemented");
    } // End Pi_perpviscous


    if (Pi_sources){//Row 8 Term 1
      TE_Pi_sources = PiSource;
      ddt(Pi) += TE_Pi_sources;
    } // End Pi_sources


    if (Pi_anomalous){
      TRACE("Ion anomalous transport");
      TE_Pi_anomalous = FCIDiv_a_Grad_perp(mul_all(a_d3d, Ti), Ne) + (2. / 3) * FCIDiv_a_Grad_perp(mul_all(a_chi3d, Ne), Ti);
      ddt(Pi) += TE_Pi_anomalous;
    } // End Pi_anomalous


    if (Pi_energyexchange){//Row 8 Term 3
      TE_Pi_energyexchange = (2. / 3) * Wi;
      ddt(Pi) += TE_Pi_energyexchange;
    } // End Pi_energyexchange


    if (parallel_sheaths){
      TRACE("Ion parallel sheaths");
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
    } // End parallel_sheaths


    if (Pi_hyper){
      TRACE("Ion pressure hyperdiffusion");
      TE_Pi_hyper = hyperdissipation(hyper_chi,Pi);
      ddt(Pi) += TE_Pi_hyper;
    } // End Pi_hyper


    if (Pi_numdiff){
      TRACE("Ion pressure numerical parallel diffusion");
      TE_Pi_numdiff = numericaldissipation(num_chi,Pi);
      ddt(Pi) += TE_Pi_numdiff;
    } // End Pi_numdiff

    
  } // End evolve_ti


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


Field3D Hermes::hyperdissipation(const Field3D &a, const Field3D &b) {
  return -a * (D4DX4(b)/SQSQ_g_11 + D4DZ4(b)/SQSQ_g_33);
  //return -a * (D4DZ4(b)/SQSQ_g_33);
}

Field3D Hermes::numericaldissipation(const Field3D &a, const Field3D &b) {
  return a * Grad2_par2(b);
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

