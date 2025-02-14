/*

0    Copyright B.Dudson, J.Leddy, University of York, 2016-2019
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
#include "neutral-model.hxx"
#include "atomicpp/ImpuritySpecies.hxx"
#include "atomicpp/Prad.hxx"

std::string parbc{"parallel_neumann_o2"};

using bout::globals::mesh;

template <typename T>
T max_abs(T a, T b) {
    return (std::abs(a) > std::abs(b)) ? a : b;
}

// Recursive case for more than two arguments                                                                                                                                                                      
template <typename T, typename... Args>
T max_abs(T first, Args... args) {
    return max_abs(first, max_abs(args...));
}


BoutReal limitFreeScale(BoutReal fm, BoutReal fc) {
  if (fm <= fc) {
    return 1; // Neumann rather than increasing into boundary
  }
  BoutReal fp = fc / fm;
  return std::max(fp, 0.98);
}


BoutReal limitFree(BoutReal fm, BoutReal fc){
  if (fc>fm){
    return fc;
  }
  BoutReal fp = fc + 0.5*(fc-fm);   // the .5 is coming from the fact, that we extralpolate to the boundary between the center cell and the up cell
  return fp;
}



BoutReal interpolate_sheathneighbour(BoutReal fc, BoutReal finterface){
  return finterface + (finterface-fc);
}

BoutReal rampfactor(BoutReal thistime, BoutReal timecut){
  if (thistime>timecut){
    return 1.0;
  } else {
    return (thistime/timecut);
  }
}









BoutReal clip(BoutReal value, BoutReal min, BoutReal max) {
  if (value < min)
    return min;
  if (value > max)
    return max;
  return value;
}

BoutReal floor(BoutReal value, BoutReal min) {
  if (value < min)
    return min;
  return value;
}

Ind3D indexAt(const Field3D& f, int x, int y, int z) {
  int ny = f.getNy();
  int nz = f.getNz();
  return Ind3D{(x * ny + y) * nz + z, ny, nz};
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
  f.yup(1).setRegion("RGN_YPAR_+2");
  f.ydown().setRegion("RGN_YPAR_-1");
  f.ydown(1).setRegion("RGN_YPAR_-2");
}

// For first order fields
const Field3D &yup(const Field3D &f) { return f.yup(); }
BoutReal yup(BoutReal f) { return f; };
const Field3D &ydown(const Field3D &f) { return f.ydown(); }
BoutReal ydown(BoutReal f) { return f; };
const BoutReal yup(BoutReal f, Ind3D i) { return f; };
const BoutReal ydown(BoutReal f, Ind3D i) { return f; };
const BoutReal &yup(const Field3D &f, Ind3D i) { return f.yup()[i]; }
const BoutReal &ydown(const Field3D &f, Ind3D i) { return f.ydown()[i]; }
BoutReal &yup(Field3D &f, Ind3D i) { return f.yup()[i]; }
BoutReal &ydown(Field3D &f, Ind3D i) { return f.ydown()[i]; }
const BoutReal &_get(const Field3D &f, Ind3D i) { return f[i]; }
BoutReal &_get(Field3D &f, Ind3D i) { return f[i]; }
BoutReal _get(BoutReal f, Ind3D i) { return f; };
BoutReal copy(BoutReal f) { return f; };

// For second order fields

const Field3D &yup2(const Field3D &f) { return f.yup(1); }
BoutReal yup2(BoutReal f) { return f; };
const Field3D &ydown2(const Field3D &f) { return f.ydown(1); }
BoutReal ydown2(BoutReal f) { return f; };
const BoutReal yup2(BoutReal f, Ind3D i) { return f; };
const BoutReal ydown2(BoutReal f, Ind3D i) { return f; };
const BoutReal &yup2(const Field3D &f, Ind3D i) { return f.yup(1)[i]; }
const BoutReal &ydown2(const Field3D &f, Ind3D i) { return f.ydown(1)[i]; }
BoutReal &yup2(Field3D &f, Ind3D i) { return f.yup(1)[i]; }
BoutReal &ydown2(Field3D &f, Ind3D i) { return f.ydown(1)[i]; }





void alloc_all(Field3D &f) {
  f.allocate();
  f.splitParallelSlices();
  f.yup().allocate();
  f.ydown().allocate();
  f.yup(1).allocate();
  f.ydown(1).allocate();
  setRegions(f);
}


#define GET_ALL(name)                                                          \
  auto *name##a = &name[Ind3D(0)];                                             \
  auto *name##b = &name.yup()[Ind3D(0)];                                       \
  auto *name##c = &name.ydown()[Ind3D(0)];                                     \
  auto *name##d = &name.yup(1)[Ind3D(0)];                                      \
  auto *name##e = &name.ydown(1)[Ind3D(0)];



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
    yup2(result, i) = op(yup2(a, i), yup2(b, i));                              \
    ydown2(result, i) = op(ydown2(a, i), ydown2(b, i));              	       \
      									       \
  }                                                                            \
  template <class B> void name##_all(Field3D &result, const B &b, Ind3D i) {   \
    result[i] = op(result[i], _get(b, i));                                     \
    yup(result, i) = op(yup(result, i), yup(b, i));                            \
    ydown(result, i) = op(ydown(result, i), ydown(b, i));                      \
    yup2(result, i) = op(yup2(result, i), yup2(b, i));                         \
    ydown2(result, i) = op(ydown2(result, i), ydown2(b, i));                   \
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
      resultd[i] = ad[i] op bd[i];					       \
      resulte[i] = ae[i] op be[i];                                             \
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
      resultd[i] = ad[i] op b;                                                 \
      resulte[i] = ae[i] op b;                                                 \
      						                               \
    }                                                                          \
    setRegions(result);                                                        \
    return result;                                                             \
  }                                                                            \
  template <class A, class B>                                                  \
  void name##_all(Field3D &result, const A &a, const B &b, Ind3D i) {          \
    result[i] = _get(a, i) op _get(b, i);                                      \
    yup(result, i) = yup(a, i) op yup(b, i);                                   \
    ydown(result, i) = ydown(a, i) op ydown(b, i);                             \
    yup2(result, i) = yup2(a, i) op yup2(b, i);				       \
    ydown2(result, i) = ydown2(a, i) op ydown2(b, i);                          \
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
      ad[i] op bd[i];                                                          \
      ae[i] op be[i];      					               \
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
      ad[i] op b;                                                              \
      ae[i] op b;                                                              \
    }                                                                          \
    return a;                                                                  \
  }                                                                            \
  template <class A, class B> void name##_all_inp(A &a, const B &b, Ind3D i) { \
    a[i] op _get(b, i);                                                        \
    yup(a, i) op yup(b, i);                                                    \
    ydown(a, i) op ydown(b, i);                                                \
    yup2(a, i) op yup2(b, i);                                                  \
    ydown2(a, i) op ydown2(b, i);                                              \
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
    yup2(result, i) = op(yup2(a, i));                                          \
    ydown2(result, i) = op(ydown2(a, i));                                      \
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
    f.yup(1)[i] = val;
    f.ydown(1)[i] = val;
  }
}
void zero_all(Field3D &f) { set_all(f, 0); }

void check_all(Field3D &f) {
  checkData(f);
  checkData(f.yup());
  checkData(f.ydown());
  checkData(f.yup(1));
  checkData(f.ydown(1));
}

void ASSERT_CLOSE_ALL(const Field3D &a, const Field3D &b) {
  BOUT_FOR(i, a.getRegion("RGN_NOY")) {
    ASSERT0(std::abs(a[i] - b[i]) < 1e-10);
    ASSERT0(std::abs(yup(a, i) - yup(b, i)) < 1e-10);
    ASSERT0(std::abs(ydown(a, i) - ydown(b, i)) < 1e-10);
    ASSERT0(std::abs(yup2(a, i) - yup2(b, i)) < 1e-10);
    ASSERT0(std::abs(ydown2(a, i) - ydown2(b, i)) < 1e-10);
  }
}

/// Modifies and returns the first argument, taking the boundary from second argument
/// This is used because unfortunately Field3D::setBoundary returns void
Field3D withBoundary(Field3D &&f, const Field3D &bndry) {
  f.setBoundaryTo(bndry);
  return f;
}


const Field3D adaptive_sourceterm(const Field3D& thisfield ,const Field3D& sourceterm, const BoutReal maximum, const BoutReal overshoot){
  Field2D averaged = DC(thisfield);
  BoutReal thismax = max(averaged);
  if (thismax>maximum){
    BoutReal ratio = floor((thismax - maximum)/overshoot,0.0);
    return sourceterm * exp(-ratio);	
  } else {
    return sourceterm;
  }
}




const Field3D new_Delp2(const Field3D& a){
  
  auto *coord = mesh->getCoordinates();
  Field3D tmp = (DDX(coord->J * coord->g11)*DDX(a) + coord->J * coord->g11 * D2DX2(a))/coord->J;
  tmp += (DDZ(coord->J * coord->g33)*DDZ(a) + coord->J * coord->g33 * D2DZ2(a))/coord->J;
  tmp += (DDX(coord->J * coord->g13)*DDZ(a) + coord->J * coord->g13 * D2DXDZ(a) *2.0 + DDZ(coord->J * coord->g13)*DDX(a))/coord->J;
  return tmp;
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
  auto& optnn = opt["Nn"];
  auto& optnnvn = opt["NnVn"];
  auto& optpn = opt["Pn"];
  auto& optneutrals = opt["Neutrals"];

  isMMS = opt["solver"]["mms"].withDefault<bool>(false);
  
  output.write("Running in MMS mode? {}\n", isMMS);
  
  OPTION(optsc, evolve_plasma, true);
  OPTION(optsc, show_timesteps, false);
  if (BoutComm::rank() != 0) {
    show_timesteps = false;
  }

  OPTION(optsc, boundarydecay, false);
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
  OPTION(optsc,output_ddt,false);
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


  // Neutrals

  evolve_neutrals = optsc["evolve_neutrals"].doc("Evolve neutrals?").withDefault<bool>(false);
  if (evolve_neutrals) {
    SOLVE_FOR(Nn);
    SOLVE_FOR(NnVn);
    EvolvingVars.add(Nn,NnVn);
    if (output_ddt) {
      SAVE_REPEAT(ddt(Nn),ddt(NnVn));
    }
  } else {
    zero_all(Nn);
    zero_all(NnVn);
    zero_all(Pn);
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
  
  use_Vi = optne["use_Vi"].doc("Use ion velocity instead of electron velocity in density equation").withDefault<bool>(false);
  
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
  NVi_supsonicdampening = optnvi["NVi_supsonicdampening"].doc("Use supersonic dampening in ion momentum").withDefault<bool>(false);

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
  Pe_dampening = optpe["Pe_dampening"].doc("Use dampening of high temperatures in electron pressure").withDefault<bool>(false);



  
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
  Vort_parflow = optvort["Vort_parflow"].doc("Use parallel ion flow in vorticity").withDefault<bool>(false);
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


  // Neutral model variables

  TE_Nn = optsc["TE_Nn"].doc("Save all terms in time evolution of neutral density").withDefault<bool>(false);
  TE_NnVn = optsc["TE_NnVn"].doc("Save all terms in time evolution of neutral momentum").withDefault<bool>(false);
  TE_Pn = optsc["TE_Pn"].doc("Save all terms in time evolution of neutral pressure").withDefault<bool>(false);
  
  
  Nn_parflow = optnn["Nn_parflow"].doc("Use neutral density parallel flow").withDefault<bool>(false);
  Nn_perpflow = optnn["Nn_perpflow"].doc("Use neutral density perpendicular flow").withDefault<bool>(false);
  Nn_sources = optnn["Nn_sources"].doc("Use neutral density source terms").withDefault<bool>(false);
  Nn_hyper = optnn["Nn_hyper"].doc("Use neutral density source terms").withDefault<bool>(false);
  
  NnVn_parflow = optnnvn["NnVn_parflow"].doc("Use neutral momentum parallel flow").withDefault<bool>(false);
  NnVn_perpflow = optnnvn["NnVn_perpflow"].doc("Use neutral momentum parallel flow").withDefault<bool>(false);
  NnVn_pargradient = optnnvn["NnVn_pargradient"].doc("Use neutral momentum parallel flow").withDefault<bool>(false);
  NnVn_pardiffusion = optnnvn["NnVn_pardiffusion"].doc("Use neutral momentum parallel flow").withDefault<bool>(false);
  NnVn_friction = optnnvn["NnVn_friction"].doc("Use neutral momentum parallel flow").withDefault<bool>(false);
  NnVn_hyper = optnnvn["NnVn_hyper"].doc("Use neutral momentum parallel flow").withDefault<bool>(false);

  
  Pn_parflow = optpn["Pn_parflow"].doc("Use neutral momentum parallel flow").withDefault<bool>(false);
  Pn_perpflow = optpn["Pn_perpflow"].doc("Use neutral momentum parallel flow").withDefault<bool>(false);
  Pn_parcompression = optpn["Pn_parcompression"].doc("Use neutral momentum parallel flow").withDefault<bool>(false);
  Pn_perpdiffusion = optpn["Pn_perpdiffusion"].doc("Use neutral momentum parallel flow").withDefault<bool>(false);
  Pn_sources = optpn["Pn_sources"].doc("Use neutral momentum parallel flow").withDefault<bool>(false);
  Pn_hyper = optpn["Pn_hyper"].doc("Use neutral momentum parallel flow").withDefault<bool>(false);
  
  TE_Nn_parflow = 0.0;
  TE_Nn_perpflow = 0.0;
  TE_Nn_sources = 0.0;
  TE_Nn_hyper = 0.0;
  
  TE_NnVn_parflow = 0.0;
  TE_NnVn_perpflow = 0.0;
  TE_NnVn_pargradient = 0.0;
  TE_NnVn_pardiffusion = 0.0;
  TE_NnVn_friction = 0.0;
  TE_NnVn_hyper = 0.0;

  
  TE_Pn_parflow = 0.0;
  TE_Pn_perpflow = 0.0;
  TE_Pn_parcompression = 0.0;
  TE_Pn_perpdiffusion = 0.0;
  TE_Pn_sources = 0.0;
  TE_Pn_hyper = 0.0;
  

  if (TE_Nn){
    SAVE_REPEAT(TE_Nn_parflow, TE_Nn_perpflow, TE_Nn_sources, TE_Nn_hyper);
  }

  if (TE_NnVn){
    SAVE_REPEAT(TE_NnVn_parflow, TE_NnVn_perpflow, TE_NnVn_pargradient, TE_NnVn_pardiffusion, TE_NnVn_friction, TE_NnVn_hyper);
  }

  if (TE_Pn){
    SAVE_REPEAT(TE_Pn_parflow, TE_Pn_perpflow, TE_Pn_parcompression, TE_Pn_perpdiffusion, TE_Pn_sources, TE_Pn_hyper);
  }

  OPTION(optneutrals,Recycling_coef, 0.95);
  OPTION(optneutrals, floor_Nn, 1e-5);
  OPTION(optneutrals, floor_Tn, 0.1/20.0);
  OPTION(optneutrals, neutralplasmainteraction, false);

  
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
  TE_NVi_supsonicdampening = 0.0;
  if (TE_NVi) {
    SAVE_REPEAT(TE_NVi_ExB, TE_NVi_mag, TE_NVi_parflow, TE_NVi_parpressure, TE_NVi_parviscos, TE_NVi_collision, TE_NVi_anomalous);
    SAVE_REPEAT(TE_NVi_hyper, TE_NVi_numdiff,TE_NVi_supsonicdampening);
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
  TE_Pe_dampening = 0.0;
  TE_Pe_sheath = 0.0;
  if (TE_Pe) {
    SAVE_REPEAT(TE_Pe_ExB, TE_Pe_mag, TE_Pe_parflow, TE_Pe_conduction, TE_Pe_ohmic, TE_Pe_thermalforce, TE_Pe_thermalcurrent);
    SAVE_REPEAT(TE_Pe_collision, TE_Pe_anomalous, TE_Pe_sources, TE_Pe_energyexchange, TE_Pe_hyper, TE_Pe_numdiff);
    SAVE_REPEAT(TE_Pe_dampening,TE_Pe_sheath);
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
  TE_Vort_parflow = 0.0;
  if (TE_Vort) {
    SAVE_REPEAT(TE_Vort_mag, TE_Vort_parcurrent, TE_Vort_polarcurrent, TE_Vort_collision, TE_Vort_parviscous, TE_Vort_anomalous);
    SAVE_REPEAT(TE_Vort_hyper, TE_Vort_numdiff,TE_Vort_parflow);
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
  OPTION(optnumerics, use_new_conduction, false);
  OPTION(optnumerics, use_new_viscosity, false);
  OPTION(optnumerics, use_new_div_par, false);
  OPTION(optnumerics, use_new_grad_par, false);
  OPTION(optnumerics, use_new_divagradperp, false);
  OPTION(optnumerics, use_Delp2, false);
  OPTION(optnumerics, use_slope_limiter, false);


  OPTION(optnumerics, use_rhie_interpolation, false);
  if (use_rhie_interpolation){
    alloc_all(rhie_cor_up);
    alloc_all(rhie_cor_down);
    SAVE_REPEAT(rhie_cor_up,rhie_cor_down);
  }

  
  OPTION(optsc, boussinesq, false);
  OPTION(optnumerics, check_finite, false);
  OPTION(optnumerics, floor_vel , -1.0);
  // Switches for different methods to support numerical stability
  
  OPTION(optnumerics, floor_kappa_ipar, -1.0);
  OPTION(optnumerics, floor_kappa_epar,-1.0);
  OPTION(optnumerics, NVi_supsonic_dissipation, false);
  OPTION(optnumerics, NVi_supsonic_factor, 1.0);
  OPTION(optnumerics, Ve_supsonic_dissipation, false);
  OPTION(optnumerics, Ve_supsonic_factor, 1.0);
  OPTION(optnumerics, Ve_supsonic_cut, 1.0);
  OPTION(optnumerics, NVi_supsonic_cut, 1.0);
  OPTION(optnumerics, Pe_dampening_Te, 8.0);
  OPTION(optnumerics, Pe_dampening_factor , 1.0);

  
  OPTION(optnumerics, ne_bndry_flux, false);
  OPTION(optnumerics, pe_bndry_flux, false);
  OPTION(optnumerics, vort_bndry_flux, false);

  OPTION(optnumerics, flux_limit_alpha, -1);
  OPTION(optnumerics, kappa_limit_alpha, -1);
  OPTION(optnumerics, eta_limit_alpha, -1);
  OPTION(optnumerics, floor_eta_epar, -1);

  
  OPTION(optnumerics, scale_ExB, 1.0);
  OPTION(optnumerics, resistivity_multiply, 1.0);
  OPTION(optnumerics, electron_weight, 1.0);
  OPTION(optnumerics, poloidal_flows, false);

  OPTION(optnumerics, Ve_supsonic_factor, 1.0);

  OPTION(optnumerics, floor_Ne,5e-2);
  OPTION(optnumerics, floor_Te,0.1);
  OPTION(optnumerics, floor_Ti,0.1);

  OPTION(optnumerics, use_Te_limiter, false);
  OPTION(optnumerics, use_Ti_limiter, false);
  OPTION(optnumerics, Te_limiter_value, 1.0);
  OPTION(optnumerics, Ti_limiter_value, 1.0);
  OPTION(optnumerics, use_Ve_limiter, false);
  OPTION(optnumerics, Ve_limiter_value, 5.0);
  
  OPTION(optnumerics, use_viscosity_limiter,false);
  OPTION(optnumerics, viscosity_limiter_value, 10.0);

  OPTION(optnumerics, use_conduction_limiter,false);
  OPTION(optnumerics, conduction_limiter_value, 1.0);


  OPTION(optnumerics, adapt_source, false);
  OPTION(optnumerics, Ne_target, 1.0);
  OPTION(optnumerics, Te_target, 1.0);
  OPTION(optnumerics, Ti_target, 1.0);
  OPTION(optnumerics, adaptive_overshoot, 1.0);


  
  
  
  // Sheath switches
  
  OPTION(optsheath, sheath_model, 0);
  OPTION(optsheath, sheath_gamma_e, 7.0);
  OPTION(optsheath, sheath_gamma_i, 3.0);
  OPTION(optsheath, sheath_infsink, false);
  OPTION(optsheath, infsink_Te, 2.0);
  OPTION(optsheath, infsink_Ne, 1.0);
  OPTION(optsheath, infsink_amp, 1.0);
  OPTION(optsheath, neutral_vwall, 1. / 3);  // 1/3rd Franck-Condon energy at wall
  OPTION(optsheath, sheath_yup, true);       // Apply sheath at yup?
  OPTION(optsheath, sheath_ydown, true);     // Apply sheath at ydown?
  OPTION(optsheath, test_boundaries, false); // Test boundary conditions
  OPTION(optsheath, parallel_sheaths, false); // Apply parallel sheath conditions?
  OPTION(optsheath, par_sheath_model, 0);
  OPTION(optsheath, par_sheath_ve, true);
  OPTION(optsheath, sheath_ramp, false);
  OPTION(optsheath, sheath_ramp_time, 1e6);
  SAVE_REPEAT(sheath_ramp_factor);
  sheath_allow_supersonic = optsheath["sheath_allow_supersonic"]
          .doc("If plasma is faster than sound speed, go to plasma velocity")
          .withDefault<bool>(true);
  
  OPTION(optsheath, sheath_interpolate, false);

  
  
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

  
  anomalous_Dn = optneutrals["anomalous_Dn"].doc("Anomalous neutral diffusion").withDefault(0.0);
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

  num_D /= (rho_s0 * rho_s0 * rho_s0 * rho_s0) * Omega_ci;
  num_nu /= (rho_s0 * rho_s0 * rho_s0 * rho_s0) * Omega_ci;
  num_chi /= (rho_s0 * rho_s0 * rho_s0 * rho_s0) * Omega_ci;

  
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

  if(anomalous_Dn > 0.0){
    anomalous_Dn /= rho_s0 * rho_s0 * Omega_ci;
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
  if (!isMMS){
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
  }

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

  Field3D Test551 = coord->J.ynext(1);
  Field3D Test552 = coord->J.ynext(2);
  Field3D Test513 = coord->g11 / coord->g13;
  Field3D Test514 = coord->g11 / coord->J;

  
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
    //logBxy.applyBoundary("neumann");
    //logBxyz.applyBoundary("neumann");
    mesh->communicate(logBxy, logBxyz);
    logBxy.applyParallelBoundary(parbc);
    logBxyz.applyParallelBoundary(parbc);
    output_info.write("Setting from log");
    coord->Bxy = exp_all(logBxy);
    Bxyz = exp_all(logBxyz);
    SAVE_ONCE(Bxyz);
    ASSERT1(min(Bxyz) > 0.0);

    mesh->communicate(Bxyz,coord->Bxy);
    
    /*
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

    */

    /*
    bout::checkPositive(coord->Bxy, "f", "RGN_NOCORNERS");
    bout::checkPositive(coord->Bxy.yup(), "fyup", "RGN_YPAR_+1");
    bout::checkPositive(coord->Bxy.ydown(), "fdown", "RGN_YPAR_-1");
    */
    logB = log(Bxyz);
    if (use_bracket){
      bracket_factor = sqrt(coord->g_22) / (coord->J * Bxyz);
    } else {
      bracket_factor = sqrt(coord->g_22) / (coord->J);
    }

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
      
      bxcvx *= rho_s0;
      bxcvy *= rho_s0;
      bxcvz *= rho_s0;
      
      
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
  
    // Use older Laplacian solver                                                                                                                 

  phiSolver = Laplacian::create(&opt["phiSolver"]);
  phiSolver->setCoefC(1./ SQ(coord->Bxy));

  
  phi = 0.0;
  Ve = 0.0;
  Vi = 0.0;
  Jpar = 0.0;

  
  
  phi.setBoundary("phi"); // For y boundaries                                                                                                     

  restart.addOnce(phi, "phi");
  if (electromagnetic){
    aparSolver = LaplaceXZ::create(mesh,&opt["aparSolver"],CELL_CENTER);
  }
  
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
  debug_decay_Ne = 0.0;
  
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
  Jpar_sheath = 0.0;
  Vort_sheath = 0.0;
  phi_sheath = 0.0;
  
  a = 0.0;
  eta_limit_denom = 0.0;
  Te_yup = 0.0;
  Te_ydown = 0.0;

  Te_ythis = 0.0;
  Te_yprev = 0.0;
  Te_ynext = 0.0;
  
  kappa_epar_yup = 0.0;
  kappa_epar_ydown = 0.0;
  debug_Pe_conduction_A = 0.0;
  debug_Pe_conduction_B = 0.0;
  debug_sheath_infsink = 0.0;
  SAVE_REPEAT(Te, Ti);
  NVi_ym2 = 0.0;
  NVi_ym1 = 0.0;
  NVi_yp1 = 0.0;
  NVi_yp2 = 0.0;
  Ne_ym2 = 0.0;
  Ne_ym1 = 0.0;
  Ne_yp1 = 0.0;
  Ne_yp2 = 0.0;
  Te_ym2 = 0.0;
  Te_ym1 = 0.0;
  Te_yp1 = 0.0;
  Te_yp2 = 0.0;
  boundary_direction = 0.0;
  
  if (verbose) {
    SAVE_REPEAT(debug_decay_Ne);
    SAVE_ONCE(boundary_direction);
    SAVE_REPEAT(Te_ythis,Te_yprev,Te_ynext);

    SAVE_REPEAT(eta_limit_denom);

    // Save additional fields
    SAVE_REPEAT(debug_soundspeed,debug_phibndry3d);
    SAVE_REPEAT(tau_e, tau_i);

    SAVE_REPEAT(NVi_ym2, NVi_ym1 , NVi_yp1, NVi_yp2);
    SAVE_REPEAT( Ne_ym2 , Ne_ym1 , Ne_yp1 , Ne_yp2, Te_ym2 , Te_ym1 , Te_yp1 , Te_yp2);
    
    SAVE_REPEAT(Jpar_sheath);

    

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
    SAVE_REPEAT(a);
  }

  zero_all(phi);
  zero_all(psi);


  for (const auto &bndry_par :
           mesh->getBoundariesPar(BoundaryParType::xout)) {
        for (const auto& pnt : *bndry_par) {
          const auto i = pnt.ind();
	  if (pnt.dir > 0.0){
	    boundary_direction[i] += 1;
	  } else if (pnt.dir < 0.0){
	    boundary_direction[i] += 10;
	  }
	}
  }

  

  
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

  alloc_all(fastest_espeed);
  alloc_all(fastest_ispeed);
  
  alloc_all(Te);
  alloc_all(Ti);
  alloc_all(Vi);
  alloc_all(a);
  alloc_all(b);
  alloc_all(d);

  alloc_all(Te32);
  alloc_all(Ti32);
  

  alloc_all(Ne);
  alloc_all(Te);
  alloc_all(Ti);
  alloc_all(Vi);
  alloc_all(Pi);
  alloc_all(Pe);

  if (evolve_neutrals){
    alloc_all(Nn);
    alloc_all(NnVn);
    alloc_all(Pn);
    alloc_all(Tn);
    alloc_all(Vn);
    SAVE_REPEAT(Tn,Vn);
    alloc_all(Sneutral);
    alloc_all(Fn);
    alloc_all(Qin);
    alloc_all(Rn);
    alloc_all(Riz);
    alloc_all(Rrc);
    alloc_all(Rcx);
    alloc_all(Recycling_flux);
    alloc_all(Dnn);
    if (verbose){
      SAVE_REPEAT(Sneutral,Fn,Rn,Qin,Riz,Rrc,Rcx,Recycling_flux);
      SAVE_REPEAT(Dnn, Pn);
    }
    
  }


  if (isMMS){
    xl = opt["xl"].withDefault(Field3D{0.0});
    yl = opt["yl"].withDefault(Field3D{0.0});
    zl = opt["zl"].withDefault(Field3D{0.0});

    SAVE_ONCE(xl,yl,zl);

  }
  set_all(oness, 1.0);
  
  // Here are some sanity checks for the flags

  if (evolve_vort && !calc_potential && !isMMS){
    throw BoutException("Evolving vorticity but not the potential");
  }


  setPrecon((preconfunc)&Hermes::precon);

  
  return 0;
}

int Hermes::rhs(BoutReal t) {
  if (show_timesteps) {
    printf("TIME = %e\r", t);
  }


  Coordinates *coord = mesh->getCoordinates();
  
  // Communicate evolving variables
  // Note: Parallel slices are not calculated because parallel derivatives
  // are calculated using field aligned quantities



  Ne.applyBoundary(t);
  NVi.applyBoundary(t);
  Pe.applyBoundary(t);
  Vort.applyBoundary(t);
  Pi.applyBoundary(t);
  VePsi.applyBoundary(t);

  if (evolve_neutrals){
    Nn.applyBoundary(t);
    NnVn.applyBoundary(t);
  }
  
  BOUT_FOR(i, Ne.getRegion("RGN_NOY")) {



    Vi[i] = NVi[i] / Ne[i];
    Te[i] = floor(Pe[i] / Ne[i],floor_Te);
    Ti[i] = floor(Pi[i] / Ne[i],floor_Ti);
    
    Ne[i] = floor(Ne[i], floor_Ne);

    NVi[i] = Ne[i] * Vi[i];
    Pe[i] = Ne[i] * Te[i];
    Pi[i] = Ne[i] * Ti[i];


    if (evolve_neutrals){
      Vn[i] = NnVn[i] / Nn[i];
      Nn[i] = floor(Nn[i], floor_Nn);
      
      NnVn[i] = Nn[i] * Vn[i];
      Pn[i] = Nn[i] * Ti[i];
    }
  }
  
  if (isMMS==false && boundarydecay==true){
    if (boundarydecay==true){
      if (mesh->lastX()) {
	int n = mesh->LocalNx;
	for (int j = mesh->ystart; j <= mesh->yend; j++) {
	  for (int k = 0; k < mesh->LocalNz; k++) {
	    // Extrapolate X-boundaries to have an exponential decay into the boundary
	    // Extrapolate Ne, Pe, Pi and NVi
	    // Ne
	    

	    BoutReal decay_Ne = limitFreeScale(abs(Ne(n - 4, j, k)) , abs(Ne(n - 3, j, k)));
	    //Ne(n - 2, j, k) = floor(Ne(n - 3, j, k) * decay_Ne,floor_Ne);
	    //Ne(n - 1, j, k) = floor(Ne(n - 3, j, k) * decay_Ne * decay_Ne,floor_Ne);
	    Ne(n - 2, j, k) = floor(Ne(n - 3, j, k),floor_Ne);
	    Ne(n - 1, j, k) = floor(Ne(n - 3, j, k),floor_Ne);
	    
	    // Pe
	    BoutReal decay_Te = limitFreeScale(abs(Te(n - 4, j, k)) , abs(Te(n - 3, j, k)));
	    Te(n - 2, j, k) = floor(Te(n - 3, j, k) * decay_Te,floor_Te);
	    Te(n - 1, j, k) = floor(Te(n - 3, j, k) * decay_Te * decay_Te,floor_Te);
	    // Pi
	    BoutReal decay_Ti = limitFreeScale(abs(Ti(n - 4, j, k)) , abs(Ti(n - 3, j, k)));
	    Ti(n - 2, j, k) = floor(Ti(n - 3, j, k) * decay_Ti,floor_Ti);
	    Ti(n - 1, j, k) = floor(Ti(n - 3, j, k) * decay_Ti * decay_Ti,floor_Ti);
	    // Vi
	    BoutReal decay_Vi = limitFreeScale(abs(Vi(n - 4, j, k)) , abs(Vi(n - 3, j, k)));
	    Vi(n - 2, j, k) = Vi(n - 3, j, k) * decay_Vi;
	    Vi(n - 1, j, k) = Vi(n - 3, j, k) * decay_Vi * decay_Vi;
	    // Ve                                                                                                                                                                                             
	    BoutReal decay_Ve = limitFreeScale(abs(Ve(n - 4, j, k)) , abs(Ve(n - 3, j, k)));
	    Ve(n - 2, j, k) = Ve(n - 3, j, k) * decay_Ve;
	    Ve(n - 1, j, k) = Ve(n - 3, j, k) * decay_Ve * decay_Ve;	
	    // Vort
	    //BoutReal decay_Vort = limitFreeScale(abs(Vort(n - 4, j, k)) , abs(Vort(n - 3, j, k)));
	    //Vort(n - 2, j, k) = Vort(n - 3, j, k) * decay_Vort;
	    //Vort(n - 1, j, k) = Vort(n - 3, j, k) * decay_Vort * decay_Vort;
	    Pi(n - 1, j, k) = Ti(n - 1, j, k) * Ne(n - 1, j, k);
	    Pi(n - 2, j, k) = Ti(n - 2, j, k) * Ne(n - 2, j, k);
	    Pe(n - 1, j, k) = Te(n - 1, j, k) * Ne(n - 1, j, k);
	    Pe(n - 2, j, k) = Te(n - 2, j, k) * Ne(n - 2, j, k);	
	    NVi(n - 1, j, k) = Vi(n - 1, j, k) * Ne(n - 1, j, k);
	    NVi(n - 2, j, k) = Vi(n - 2, j, k) * Ne(n - 2, j, k);        
	    VePsi(n - 1, j, k) = Ve(n - 1, j, k) - Vi(n - 1, j, k);
	    VePsi(n - 2, j, k) = Ve(n - 2, j, k) - Vi(n - 2, j, k);	  	
	  }
	}
      }
    } else {
      if (mesh->lastX()) {
        int n = mesh->LocalNx;
        for (int j = mesh->ystart; j <= mesh->yend; j++) {
          for (int k = 0; k < mesh->LocalNz; k++) {
	    Ne(n - 1, j, k) = Ne(n - 2, j, k);
	    Pe(n - 1, j, k) = Pe(n - 2, j, k);
	    Pi(n - 1, j, k) = Pi(n - 2, j, k);
	    NVi(n - 1, j, k) = NVi(n - 2, j, k);
	    Vort(n - 1, j, k) = Vort(n - 2, j, k);
	    VePsi(n - 1, j, k) = VePsi(n - 2, j, k);
	    
          }
        }
      }

    }

  } // End
  
  
  
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

  if (evolve_neutrals){
    Nn.applyParallelBoundary(parbc);
    NnVn.applyParallelBoundary(parbc);
  }



  
  
  Field3D sound_speed;
  alloc_all(sound_speed);

  BOUT_FOR(i, Ne.getRegion("RGN_ALL")) {

    div_all(Te, Pe, Ne, i);
    div_all(Vi, NVi, Ne, i);
    div_all(Ti, Pi, Ne, i);

    floor_all(Ne, floor_Ne, i);
    floor_all(Te, floor_Te, i);
    floor_all(Ti, floor_Ti, i);

    mul_all(Pe, Te, Ne, i);
    mul_all(Pi, Ti, Ne, i);
    mul_all(NVi, Vi, Ne, i);

    if(evolve_neutrals){
      div_all(Vn, NnVn, Nn, i);
      
      floor_all(Nn, floor_Nn, i);
      mul_all(Pn, Ti, Nn, i);
      mul_all(NnVn, Vn, Nn, i);
      
    }
    

    sound_speed[i] =  sqrt(Te[i] + Ti[i] * (5. / 3));
  }
  
  sound_speed.applyBoundary("neumann");

  
  fastest_ispeed = sound_speed;
  if (electromagnetic){
    fastest_espeed = sound_speed;
  } else {
    fastest_espeed = sound_speed;
  }
  


  
  if(verbose){
    debug_soundspeed = sound_speed;
  }
  
  // Set radial boundary conditions on Te, Ti, Vi
  //


  

  if (!evolve_ti){
    Pi=Pe;
  }



  //////////////////////////////////////////////////////////////
  // Calculate electrostatic potential phi

  TRACE("Electrostatic potential");
  if (calc_potential){
    Field3D phi_boundary3d;
    phi_boundary3d = 0.0;
  

    if (boussinesq) {
      if (!isMMS){
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
      } else if (isMMS){

	if (mesh->firstX()) {
          for (int j = mesh->ystart; j <= mesh->yend; j++) {
            for (int k = 0; k < mesh->LocalNz; k++) {
              phi_boundary3d(mesh->xstart - 1, j, k) = 0.5 * ( Pi(mesh->xstart - 1, j, k) + Pi(mesh->xstart, j, k));
            }
          }
        }	


        if (mesh->lastX()) {
          for (int j = mesh->ystart; j <= mesh->yend; j++) {
            for (int k = 0; k < mesh->LocalNz; k++) {
              phi_boundary3d(mesh->xend + 1, j, k) = 0.5 * ( Pi(mesh->xend + 1, j, k) + Pi(mesh->xend, j, k) );
            }
          }
        }

	

      }
      ////////////////////////////////////////////
      // Boussinesq, non-split
      // Solve all components using X-Z solver
      
      
	// Use older Laplacian solver
	// phiSolver->setCoefC(1./SQ(coord->Bxy)); // Set when initialised
      mesh->communicate(phi_boundary3d);
      phi = phiSolver->solve(mul_all(Vort , mul_all(coord->Bxy, coord->Bxy)), phi_boundary3d);//_boundary3d);
	
      
      // Hot ion term in vorticity
      debug_phibndry3d = phi_boundary3d;
      //phi.applyBoundary("neumann");
      mesh->communicate(phi);
      phi.applyParallelBoundary(parbc);
      
      phi = sub_all(phi, Pi);

      // Set the potential manually at the last cell to keep interpolation intact
      
      if (mesh->lastX()) {
	int n = mesh->LocalNx;
	for (int j = mesh->ystart; j <= mesh->yend; j++) {
	  for (int k = 0; k < mesh->LocalNz; k++) {
	    phi(n - 1, j, k) = phi(n - 2, j, k);
	  }
	}
      }
      
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

      Field3D ones = 1.0;
      Field3D tmp = -Ne*0.5*beta_e*mi_me;
      // With laplacian
      //aparSolver->setCoefD(1.0);
      //aparSolver->setCoefA(-Ne*0.5*beta_e*mi_me);
      aparSolver->setCoefs(ones,tmp);
      //psi = aparSolver->solve(-VePsi,psi);
      psi = aparSolver->solve(-VePsi*Ne,ones);
      mesh->communicate(psi);
      
      psi.applyParallelBoundary(parbc);
      
      Ve = VePsi - 0.5 * beta_e * mi_me * psi + Vi;
      if (!isMMS){
	Ve.applyBoundary("neumann");
      }
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

      sheath_ramp_factor = rampfactor(t,sheath_ramp_time);
      sheath_dpe = 0.0;
      sheath_dpi = 0.0;
      Recycling_flux = 0.0;
      for (const auto &bndry_par :
           mesh->getBoundariesPar(BoundaryParType::xout)) {
	for (const auto& pnt : *bndry_par) {
	  const auto i = pnt.ind();
	  // This if statement catech double boundaries
	  // And ignores boundaries in the negative direction, only taking the positive one
	  if (boundary_direction[i] > 10.9 && boundary_direction[i] < 11.1 && pnt.dir < 0.0);
	  else{
	    /*
	  BoutReal decay_Ne = limitFreeScale(pnt.yprev(Ne),pnt.ythis(Ne));
	  BoutReal decay_Te = limitFreeScale(pnt.yprev(Te),pnt.ythis(Te));
	  BoutReal decay_Ti = limitFreeScale(pnt.yprev(Ti),pnt.ythis(Ti));
	  if (sheath_interpolate){
	    //pnt.ynext(Ne) = floor(pnt.ythis(Ne)*decay_Ne, floor_Ne);
	    pnt.ynext(Ne) = floor(pnt.ythis(Ne), floor_Ne); // Not for Ne, sothat NVi does not increase if vi is constant
            pnt.ynext(Te) = floor(pnt.ythis(Te)*decay_Te, floor_Te);
            pnt.ynext(Ti) = floor(pnt.ythis(Ti)*decay_Ti, floor_Ti);
	    
	  } else {
	    pnt.ynext(Ne) = pnt.ythis(Ne);
            pnt.ynext(Ti) = pnt.ythis(Ti);
            pnt.ynext(Te) = pnt.ythis(Te);
	  }
	    */
	  pnt.ynext(Ne) = floor(pnt.ythis(Ne), floor_Ne); // Not for Ne, sothat NVi does not increase if vi is constant                                                                                                                                                         
	  pnt.ynext(Te) = floor(pnt.ythis(Te), floor_Te);
	  pnt.ynext(Ti) = floor(pnt.ythis(Ti), floor_Ti);

	  
	  pnt.ynext(Pi) = pnt.ynext(Ne)*pnt.ynext(Ti);
	  pnt.ynext(Pe) = pnt.ynext(Ne)*pnt.ynext(Te);

	  
	  TRACE("Sheath offset==2, interpolate sheath values");

	  BoutReal nesheath = 0.0;
	  BoutReal tesheath = 0.0;
	  BoutReal tisheath = 0.0;


	  nesheath = pnt.ythis(Ne);
	  tesheath = pnt.ythis(Te);
	  tisheath = pnt.ythis(Ti);
	  

	  BoutReal phisheath = log(sqrt(tesheath / (tesheath + tisheath))) * tesheath;
          pnt.ynext(phi) = interpolate_sheathneighbour(pnt.ythis(phi),phisheath);

	  BoutReal visheath = 0.0;
	  if (!sheath_ramp){
	    visheath = pnt.dir * sqrt((5.0/3.0)*tisheath + tesheath);
	  } else {
	    visheath = sheath_ramp_factor * (pnt.dir * sqrt((5.0/3.0)*tisheath + tesheath));
	  }

	  if (pnt.dir > 0.99 && pnt.dir < 1.01){
	    if (pnt.ythis(Vi) > visheath){
	      visheath = pnt.ythis(Vi);
	    }
	  } else {
	    if (pnt.ythis(Vi) < visheath){
	      visheath = pnt.ythis(Vi);
	    }
	  }

	  

	  BoutReal vesheath = 0.0;
	  if (evolve_vepsi){
	    if (!sheath_ramp){
	      vesheath = pnt.dir * sqrt(tesheath) * (sqrt(mi_me) / (2. * sqrt(PI))) * exp(-(phisheath/tesheath));
	    } else {
	      vesheath = sheath_ramp_factor * (pnt.dir * sqrt(tesheath) * (sqrt(mi_me) / (2. * sqrt(PI))) * exp(-(phisheath/tesheath))); 
	    }

	    if (pnt.dir > 0.99 && pnt.dir < 1.01){
	      if (pnt.ythis(Ve) > vesheath){
		vesheath = pnt.ythis(Ve);
	      }
	    } else {
	      if (pnt.ythis(Ve) < vesheath){
		vesheath = pnt.ythis(Ve);
	      }
	    }
	    

	    
	  } else {
	    vesheath = visheath;
	  }
	  

	  
	  
	  const BoutReal jsheath = nesheath * (visheath - vesheath);
	  const BoutReal nvisheath = nesheath * visheath;

	  if (sheath_interpolate){
	    pnt.ynext(Vi) = interpolate_sheathneighbour(pnt.ythis(Vi), visheath);
	    pnt.ynext(Ve) = interpolate_sheathneighbour(pnt.ythis(Ve), vesheath);
	    pnt.ynext(Jpar) = interpolate_sheathneighbour(pnt.ythis(Jpar), jsheath);
	    pnt.ynext(NVi) = interpolate_sheathneighbour(pnt.ythis(NVi), nvisheath);
	    pnt.ynext(Vort) = pnt.ythis(Vort);
	  } else {
	    pnt.ynext(Vi) = visheath;
	    pnt.ynext(Ve) = vesheath;
	    pnt.ynext(Jpar) = jsheath;
	    pnt.ynext(NVi) = nvisheath;
	    pnt.ynext(Vort) = pnt.ythis(Vort);
	  }

	  if (verbose){
	    Te_ythis[i] = pnt.ythis(Te);
	    Te_ynext[i] = pnt.ynext(Te);
	    Te_yprev[i] = pnt.yprev(Te);
	    debug_visheath[i] = visheath;
	    debug_vesheath[i] = vesheath;
	    debug_phisheath[i] = phisheath;
	    Te_sheath[i] = tesheath;
	  }
	  
	  if (abs(pnt.offset())==1){	              	   	    
	    

	    TRACE("Sheath offset==1, sheath power calculation");

            const BoutReal q_e = floor( (sheath_gamma_e - 1.5) * tesheath * nesheath * vesheath * pnt.dir , 0.0);                                                                                         
            const BoutReal flux_e = q_e * coord->J[i] / sqrt(coord->g_22[i]);
	    BoutReal power_e = 0.0;
                                                                                                                                                                                                          
            const BoutReal q_i = floor( (sheath_gamma_i - 1.0) * tisheath * nesheath * visheath * pnt.dir , 0.0);                                                                                         
            const BoutReal flux_i = q_i * coord->J[i] / sqrt(coord->g_22[i]);
	    BoutReal power_i = 0.0;

	    if (!sheath_ramp){
	      power_e = flux_e / (coord->dy[i] * coord->J[i]);
	      power_i = flux_i / (coord->dy[i] * coord->J[i]);
	    } else {
	      power_e = sheath_ramp_factor * (flux_e / (coord->dy[i] * coord->J[i]));
              power_i = sheath_ramp_factor * (flux_i / (coord->dy[i] * coord->J[i]));
	    }
	    
	    sheath_dpi[i] -= (3.0/2.0) * power_i;                                                                                                                                                         
	    sheath_dpe[i] -= (3.0/2.0) * power_e;

	    if (evolve_neutrals && Recycling_coef>0.0){
	      Recycling_flux[i] = Recycling_coef*abs(visheath * nesheath) * coord->J[i]/( sqrt(coord->g_22[i])*coord->dy[i]*coord->J[i]);
	    }
	    
	    // Also set the values in the interpolated value after the sheath, here neumann

	    TRACE("Sheath offset==1, set double next fields");

	    const int offset_factor = 1;
	    if(sheath_interpolate){	      
	      //pnt.getAt<false>(Ne, offset_factor) = floor(pnt.ythis(Ne)*decay_Ne*decay_Ne, floor_Ne);
	      pnt.getAt<false>(Ne, offset_factor) = floor(pnt.ynext(Ne), floor_Ne);
	      pnt.getAt<false>(Te, offset_factor) = floor(pnt.ynext(Te), floor_Te);
	      pnt.getAt<false>(Ti, offset_factor) = floor(pnt.ynext(Ti), floor_Ti);
	      pnt.getAt<false>(Pe, offset_factor) = pnt.getAt<false>(Ne, offset_factor) * pnt.getAt<false>(Te, offset_factor);
	      pnt.getAt<false>(Pi, offset_factor) = pnt.getAt<false>(Ne, offset_factor) * pnt.getAt<false>(Ti, offset_factor);

	      pnt.getAt<false>(phi, offset_factor) = interpolate_sheathneighbour(pnt.ythis(phi),pnt.ynext(phi));
	      pnt.getAt<false>(Vi, offset_factor) = interpolate_sheathneighbour(pnt.ythis(Vi),pnt.ynext(Vi));
	      pnt.getAt<false>(Ve, offset_factor) = interpolate_sheathneighbour(pnt.ythis(Ve),pnt.ynext(Ve));
	      pnt.getAt<false>(Jpar, offset_factor) = interpolate_sheathneighbour(pnt.ythis(Jpar),pnt.ynext(Jpar));
	      pnt.getAt<false>(NVi, offset_factor) = interpolate_sheathneighbour(pnt.ythis(NVi),pnt.ynext(NVi));
	      pnt.getAt<false>(Vort, offset_factor) = interpolate_sheathneighbour(pnt.ythis(Vort),pnt.ynext(Vort));
	    } else {       	    
	      pnt.getAt<false>(Ne, offset_factor) = pnt.ynext(Ne);
	      pnt.getAt<false>(Te, offset_factor) = pnt.ynext(Te);
	      pnt.getAt<false>(Pe, offset_factor) = pnt.ynext(Pe);
	      pnt.getAt<false>(Ti, offset_factor) = pnt.ynext(Ti);
	      pnt.getAt<false>(Pi, offset_factor) = pnt.ynext(Pi);
	    
	      pnt.getAt<false>(phi, offset_factor) = pnt.ynext(phi);
	      pnt.getAt<false>(Vi, offset_factor) = pnt.ynext(Vi);
	      pnt.getAt<false>(Ve, offset_factor) = pnt.ynext(Ve);
	      pnt.getAt<false>(Jpar, offset_factor) = pnt.ynext(Jpar);
	      pnt.getAt<false>(NVi, offset_factor) = pnt.ynext(NVi);
	      pnt.getAt<false>(Vort, offset_factor) = pnt.ynext(Vort);
	    }
	  
	  } // End interpolate_sheathneighbour
	  }
	  
	} // End for (const auto& pnt : region)
      } // End iter_regions([&](auto& region)

	  
      break;
    } // End case 0
    case 1:{
      // This is meant to conserve kinetic energy  at the sheath boundary
      // Temperature is also maintained
      // Density is this changed to account for the change in momentum
      sheath_ramp_factor = rampfactor(t,sheath_ramp_time);
      sheath_dpe = 0.0;
      sheath_dpi = 0.0;
      Recycling_flux = 0.0;
      for (const auto &bndry_par :
           mesh->getBoundariesPar(BoundaryParType::xout)) {
        for (const auto& pnt : *bndry_par) {
          const auto i = pnt.ind();
          // This if statement catech double boundaries                                                                                              
          // And ignores boundaries in the negative direction, only taking the positive one                                                          
          if (boundary_direction[i] > 10.9 && boundary_direction[i] < 11.1 && pnt.dir < 0.0);
          else{

	    BoutReal kin_E = pnt.ythis(Ne) * pnt.ythis(Vi) * pnt.ythis(Vi);
	    BoutReal tesheath = pnt.ythis(Te);
	    BoutReal tisheath = pnt.ythis(Ti);
	    BoutReal phisheath = log(sqrt(tesheath / (tesheath + tisheath))) * tesheath;
	    pnt.ynext(phi) = interpolate_sheathneighbour(pnt.ythis(phi),phisheath);

	    
	    BoutReal visheath = 0.0;
	    if (!sheath_ramp){
	      visheath = pnt.dir * sqrt((5.0/3.0)*tisheath + tesheath);
	    } else {
	      visheath = sheath_ramp_factor * (pnt.dir * sqrt((5.0/3.0)*tisheath + tesheath));
	    }
	    BoutReal vesheath = 0.0;
	    if (evolve_vepsi){
	      if (!sheath_ramp){
		vesheath = pnt.dir * sqrt(tesheath) * (sqrt(mi_me) / (2. * sqrt(PI))) * exp(-(phisheath/tesheath));
	      } else {
		vesheath = sheath_ramp_factor * (pnt.dir * sqrt(tesheath) * (sqrt(mi_me) / (2. * sqrt(PI))) * exp(-(phisheath/tesheath)));
	      }
	    } else {
	      vesheath = visheath;
	    }

	    BoutReal nesheath = floor(kin_E / (visheath*visheath), floor_Ne);
	    const BoutReal jsheath = nesheath * (visheath - vesheath);
	    const BoutReal nvisheath = nesheath * visheath;
	    

	    
	  } // End if (boundary_direction[i]

	} // End for (const auto& pnt : *bndry_par)
	
      } // End for (const auto &bndry_par
      
      
      
      break;
    } // End case 1
      
    default: {
      throw BoutException("Not implemented");
      break;
    }
    }
  }




  //////////////////////////////////////////////////////////////
  // Neutral calculations

  /*
  void NeutralModel::neutral_rates(
    const Field3D &Ne, const Field3D &Te, const Field3D &Ti,
    const Field3D &Vi, // Plasma quantities                                                                                                                                                                                                                                       
    const Field3D &Nn, const Field3D &Tn, const Field3D &Vnpar, // Neutral gas                                                                                                                                                                                                    
    Field3D &S, Field3D &F, Field3D &Qi, Field3D &R, // Transfer rates                                                                                                                                                                                                            
    Field3D &Riz, Field3D &Rrc, Field3D &Rcx,
    BoutReal NormT, BoutReal NormN, BoutReal NormB, BoutReal NormL, BoutReal NormF,
    bool ionizationloss)
  */
  
  if (evolve_neutrals){
    fci_neutral_rates( Ne, Te, Ti, Vi, Nn, Ti, Vn, Sneutral, Fn, Qin, Rn, Riz, Rrc, Rcx, Tnorm, Nnorm, Bnorm, rho_s0, Omega_ci, true, Dnn);
    Fn.applyBoundary("neumann");
    Qin.applyBoundary("neumann");
    Rn.applyBoundary("neumann");
    Riz.applyBoundary("neumann");
    Rcx.applyBoundary("neumann");
    Rrc.applyBoundary("neumann");
    Dnn.applyBoundary("neumann");
    
    mesh->communicate(Fn, Qin, Rn, Riz, Rcx, Rrc, Dnn);
    Fn.applyParallelBoundary(parbc);
    Qin.applyParallelBoundary(parbc);
    Rn.applyParallelBoundary(parbc);
    Riz.applyParallelBoundary(parbc);
    Rrc.applyParallelBoundary(parbc);
    Rcx.applyParallelBoundary(parbc);
    Dnn.applyParallelBoundary(parbc);
  }


  //////////////////////////////////////////////////////////////
  // Debug variables output
  
  if (verbose){
    BOUT_FOR(i, Ne.getRegion("RGN_NOBNDRY")){
      const auto iyp = i.yp();
      const auto iym = i.ym();
      const auto iypp = i.ypp();
      const auto iymm = i.ymm();
      NVi_ym2[i] = NVi.ydown(1)[iymm];
      NVi_ym1[i] = NVi.ydown()[iym];
      NVi_yp2[i] = NVi.yup(1)[iypp];
      NVi_yp1[i] = NVi.yup()[iyp];
      Te_ym2[i] = Te.ydown(1)[iymm];
      Te_ym1[i] = Te.ydown()[iym];
      Te_yp2[i] = Te.yup(1)[iypp];
      Te_yp1[i] = Te.yup()[iyp];
      Ne_ym2[i] = Ne.ydown(1)[iymm];
      Ne_ym1[i] = Ne.ydown()[iym];
      Ne_yp2[i] = Ne.yup(1)[iypp];
      Ne_yp1[i] = Ne.yup()[iyp];
    }
  }

  
  //////////////////////////////////////////////////////////////
  // Plasma quantities calculated.
  // At this point we have calculated all boundary conditions,
  // and auxilliary variables like jpar, phi, psi


  Te32= mul_all(Te,sqrt_all(Te));

  Ti32= mul_all(Ti,sqrt_all(Ti));

  
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
  
  // Electron parallel viscosity
  if (!use_new_viscosity){
    eta_epar = mul_all(0.7333, mul_all(mi_me,mul_all(tau_e,Pe)));
  } else {
    eta_epar = mul_all(div_all(4.0,3.0),mul_all(0.73,mul_all(Pe,tau_e)));
  }
    
  if (eta_limit_alpha>0.0){
    Field3D qm_cl = eta_epar * Grad_par(Ve);
    mesh->communicate(qm_cl);
    qm_cl.applyParallelBoundary(parbc);
    
    Field3D qm_fl = mul_all(eta_limit_alpha,mul_all(Pe,mi_me));
    Field3D tmp = abs(div_all(qm_cl,qm_fl));
    mesh->communicate(tmp);
    tmp.applyParallelBoundary(parbc);
    eta_limit_denom = add_all(1,tmp);
    eta_epar = div_all(eta_epar,eta_limit_denom);
      
  }

  if (floor_eta_epar>0.0){
    BOUT_FOR(i, eta_epar.getRegion("RGN_ALL")) {
      floor_all(eta_epar, floor_eta_epar, i);
    }
  }
  
  //////////////////////////////////////////////////////////////                                                                        
  TRACE("Calculating resistivity");

  //nu = resistivity_multiply / (1.96 * tau_e * mi_me);
  nu = div_all(resistivity_multiply,mul_all(1.96,mul_all(tau_e,mi_me)));

  Wi = mul_all(div_all(3.0,mi_me),mul_all(Ne,div_all(sub_all(Te,Ti),tau_e)));


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
      if(!use_Vi){
	if (!use_new_div_par){
	  Field3D neve = mul_all(Ne,Ve);
	  TE_Ne_parflow = -Div_par(neve);
	} else if(use_rhie_interpolation){
	  TE_Ne_parflow = -Div_par_rhie(Ne, Ve, add_all(Pe,Pi), rhie_cor_up, rhie_cor_down);
	} else {
	  TE_Ne_parflow = -Div_par_mod(Ne,Ve,fastest_espeed, use_slope_limiter);
	}

      } else {
	if (!use_new_div_par){
	  Field3D nevi = mul_all(Ne,Vi);
	  TE_Ne_parflow = -Div_par(nevi);
	} else if (use_rhie_interpolation){
	  TE_Ne_parflow = -Div_par_rhie(Ne, Vi, add_all(Pe,Pi), rhie_cor_up, rhie_cor_down);
	} else {
	  TE_Ne_parflow = -Div_par_mod(Ne,Vi,fastest_ispeed, use_slope_limiter);
	}
      }
      ddt(Ne) += TE_Ne_parflow;
    }  // End Ne_parflow

    
    if (Ne_collision){// Row 3
      TRACE("Density collisions");
      throw BoutException("Density collisions not implemented");
    }  // End Ne_collision

    
    if (Ne_anomalous){// Row 4 
      TRACE("Density anomalous");
      if (use_new_divagradperp){
	TE_Ne_anomalous = Div_a_Grad_perp_mod(a_d3d, Ne);
      } else {
	TE_Ne_anomalous = Div_a_Grad_perp_curv(a_d3d, Ne);
	//TE_Ne_anomalous = a_d3d * new_Delp2(Ne); 
      }
      ddt(Ne) += TE_Ne_anomalous;
    }  // End Ne_anomalous

    
    if (Ne_sources){//Row 5 Term 2
      TRACE("Density sources");
      if (adapt_source) {
	TE_Ne_sources = adaptive_sourceterm(Ne ,NeSource, Ne_target, adaptive_overshoot);
      } else {
	TE_Ne_sources = NeSource;
      }

      
      if (evolve_neutrals && neutralplasmainteraction){
	TE_Ne_sources -= Sneutral;
      }
      
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
      if (!use_new_div_par){
	TE_Vort_parcurrent = Div_par(Jpar);
      } else {
	TE_Vort_parcurrent = Div_par_mod(Ne, sub_all(Vi,Ve),fastest_ispeed, use_slope_limiter);
      }

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
    

    if (Vort_anomalous){
      TRACE("Vort anomalous");
      if (!use_new_divagradperp){
        TE_Vort_anomalous = Div_a_Grad_perp_curv(a_nu3d, Vort);
      } else {
        TE_Vort_anomalous = Div_a_Grad_perp_mod(a_nu3d, Vort);
      }
      ddt(Vort) += TE_Vort_anomalous;
    }



    
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


    if (Vort_parflow){
      TRACE("Vorticity parallel flow");
      if (!use_new_div_par){
	Field3D VortVi = mul_all(Vort,Vi);
	TE_Vort_parflow = -Div_par(VortVi);
      } else {
	TE_Vort_parflow = -Div_par_mod(Vort,Vi,fastest_ispeed, use_slope_limiter);
      }    
      ddt(Vort) += TE_Vort_parflow;
    }

    
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
      if (!use_new_grad_par){
	TE_VePsi_parefield = mi_me * Grad_par(phi);
      } else {
	TE_VePsi_parefield = mi_me * Grad_par_mod(phi);
      }
      ddt(VePsi) += TE_VePsi_parefield;
    } //End VePsi_parefield

    
    if (VePsi_parpressure){//Row 1 Term 2
      if (!use_new_grad_par){
	TE_VePsi_parpressure = -mi_me * Grad_par(Pe) / Ne;
      } else {
	TE_VePsi_parpressure = -mi_me * Grad_par_mod(Pe) / Ne;
      }
      ddt(VePsi) += TE_VePsi_parpressure;
    } //End VePsi_parpressure


    if (VePsi_partemp){//Row 1 Term 3
      if (!use_new_grad_par){
	TE_VePsi_partemp = -mi_me * 0.71 * Grad_par(Te);
      } else {
	TE_VePsi_partemp = -mi_me * 0.71 * Grad_par_mod(Te);
      }
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
      if(!use_Vi){
	if(!use_new_div_par){
	  TE_VePsi_parflow = -Ve * Div_par(sub_all(Ve,Vi));
	} else {
	  TE_VePsi_parflow = -Ve * Div_par(sub_all(Ve,Vi));
	}
      } else {
	if(!use_new_div_par){
	  TE_VePsi_parflow = -Vi * Div_par(sub_all(Ve,Vi));
	} else {
	  TE_VePsi_parflow = -Vi * Div_par(sub_all(Ve,Vi));
	}       
      }
      ddt(VePsi) += TE_VePsi_parflow;
    } // End VePsi_parflow

    

    if (VePsi_hyper){
      TRACE("VePsi hyperdiffusion");
      TE_VePsi_hyper = hyperdissipation(hyper_nu,Ve);
      ddt(VePsi) += TE_VePsi_hyper;
    } // End VePsi_hyper


    if (VePsi_numdiff){
      TRACE("VePsi numerical parallel diffusion");
      TE_VePsi_numdiff = numericaldissipation(num_nu,Ve);
      ddt(VePsi) += TE_VePsi_numdiff;
    } // End VePsi_numdiff


    if (VePsi_parallelvisc){
      TRACE("VePsi parallel viscosity");
      
      if(!use_new_conduction){
	TE_VePsi_parallelvisc = Div_par_K_Grad_par(eta_epar,Ve);
      } else {
	TE_VePsi_parallelvisc = Div_par_K_Grad_par_mod(eta_epar,Ve);
      }

      if (use_viscosity_limiter){
	TE_VePsi_parallelvisc = term_limiter(TE_VePsi_parallelvisc, viscosity_limiter_value);
      }
      ddt(VePsi) += TE_VePsi_parallelvisc; 
    } // End VePsi_parallelvisc


    if (VePsi_supsonicdampening){
      TE_VePsi_supsonicdampening = 0.0;
      BOUT_FOR(i, VePsi.getRegion("RGN_NOBNDRY")){
	if(Ve[i] < (-Ve_supsonic_cut*sqrt(mi_me)*sound_speed[i])){
	  //BoutReal tmp = abs(Ve[i]) - Ve_supsonic_cut*sqrt(mi_me)*sound_speed[i];
	  BoutReal tmp = abs(Ve[i])/(Ve_supsonic_cut*sqrt(mi_me)*sound_speed[i]);
	  TE_VePsi_supsonicdampening[i] = Ve_supsonic_factor * (floor(exp(tmp)-1.0,0.0));
	} else if (Ve[i] > (Ve_supsonic_cut*sqrt(mi_me)*sound_speed[i])){
	  //BoutReal tmp = abs(Ve[i]) - Ve_supsonic_cut*sqrt(mi_me)*sound_speed[i];
	  BoutReal tmp = abs(Ve[i])/(Ve_supsonic_cut*sqrt(mi_me)*sound_speed[i]);
	  TE_VePsi_supsonicdampening[i] = -Ve_supsonic_factor * (floor(exp(tmp)-1.0,0.0));
	}
      }
      ddt(VePsi) += TE_VePsi_supsonicdampening;      
    } // End VePsi_supsonicdampening

    if (VePsi_anomalous){
      TRACE("VePsi anomalous");
      if (!use_new_divagradperp){
	TE_VePsi_anomalous = Div_a_Grad_perp_curv(a_nu3d, Ve);
      } else {
	TE_VePsi_anomalous = Div_a_Grad_perp_mod(a_nu3d, Ve);
      }
      ddt(VePsi) += TE_VePsi_anomalous;
    }

        
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

      if(!use_new_div_par){
	auto nvivi = mul_all(NVi,Vi);
	TE_NVi_parflow = -Div_par(nvivi);
      } else if (use_rhie_interpolation){
	TE_NVi_parflow = -Div_par_rhie(NVi, Vi, add_all(Pe,Pi), rhie_cor_up, rhie_cor_down);
      } else {
	TE_NVi_parflow = -Div_par_nvv_mod(Ne,Vi,fastest_ispeed);
      }
      
      ddt(NVi) += TE_NVi_parflow;
    } // End NVi_parflow

    
    if (NVi_parpressure){//Row 2
      Field3D peppi = add_all(Pe, Pi);
      if(!use_new_grad_par){
	TE_NVi_parpressure = -Grad_par(peppi);
      } else {
	TE_NVi_parpressure = -Grad_par_mod(peppi);
      }
      ddt(NVi) += TE_NVi_parpressure;
    } // End NVi_parpressure


    if (NVi_parviscos){//Row 3
      Field3D tmp = 0.0;
      if(!use_new_conduction){
	tmp = Div_par_K_Grad_par(div_all(mul_all(Pi,tau_i),coord->Bxy),mul_all(B12,Vi));
      } else {
	tmp = Div_par_K_Grad_par_mod(div_all(mul_all(Pi,tau_i),coord->Bxy),mul_all(B12,Vi));
      }
      TE_NVi_parviscos = 1.28*B12*tmp;
      ddt(NVi) += TE_NVi_parviscos;
    } // End NVi_parviscos


    if (NVi_collision){
      throw BoutException("NVi collisions not implemented!");
    }

    
    if (NVi_anomalous){//Row 5

      if (use_Delp2){
	//TE_NVi_anomalous = a_d3d * Vi * new_Delp2(Ne) + a_nu3d * Ne * new_Delp2(Vi);
	TE_NVi_anomalous = a_nu3d * Ne * new_Delp2(Vi);
      } else {
	if (!use_new_divagradperp){
	  TE_NVi_anomalous = Div_a_Grad_perp_curv(mul_all(Vi, a_d3d), Ne);
	  TE_NVi_anomalous += Div_a_Grad_perp_curv(mul_all(Ne, a_nu3d), Vi);
	} else {
	  
	  TE_NVi_anomalous = Div_a_Grad_perp_mod(mul_all(Vi,a_d3d), Ne);
	  TE_NVi_anomalous += Div_a_Grad_perp_mod(mul_all(Ne, a_nu3d), Vi);
	  
	}
      }
      ddt(NVi) += TE_NVi_anomalous;
    }


    if (NVi_hyper){
      TRACE("Ion momentum hyperdiffusion");
      TE_NVi_hyper = hyperdissipation(hyper_nu,Vi);
      ddt(NVi) += TE_NVi_hyper;
    } // End NVi_hyper


    if (NVi_numdiff){
      TRACE("Ion momentum numerical parallel diffusion");
      TE_NVi_numdiff = numericaldissipation(num_nu,NVi);
      ddt(NVi) += TE_NVi_numdiff;
    } // End NVi_numdiff


    if (NVi_supsonicdampening){
      TE_NVi_supsonicdampening = 0.0;
      BOUT_FOR(i, NVi.getRegion("RGN_NOBNDRY")){
        if(Vi[i] < (-NVi_supsonic_cut*sound_speed[i])){
          BoutReal tmp = abs(Vi[i])/(NVi_supsonic_cut*sound_speed[i]);
          TE_NVi_supsonicdampening[i] = NVi_supsonic_factor * (floor(exp(tmp)-1.0,0.0));
        } else if (Vi[i] > (NVi_supsonic_cut*sound_speed[i])){
          BoutReal tmp = abs(Vi[i])/(NVi_supsonic_cut*sound_speed[i]);
          TE_NVi_supsonicdampening[i] = -NVi_supsonic_factor * (floor(exp(tmp)-1.0,0.0));
	}
      }
      ddt(NVi) += TE_NVi_supsonicdampening;
    } // End NVi_supsonicdampening


    if (evolve_neutrals && neutralplasmainteraction){
      ddt(NVi) -= (Vi - Vn)  * (Rrc + Rcx);
    }



    
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
      if(!use_new_div_par){
	Field3D peve = mul_all(Pe,Ve);
	TE_Pe_parflow = -Div_par(peve) - (2. / 3) * Pe * Div_par(Ve);
      } else if(use_rhie_interpolation){
	TE_Pe_parflow = -Div_par_rhie(Pe, Ve, add_all(Pe,Pi), rhie_cor_up,rhie_cor_down ) -
	  2.0/3.0 * Pe * Div_par_rhie(oness, Ve, add_all(Pe,Pi), rhie_cor_up, rhie_cor_down);
      } else {
	TE_Pe_parflow = -Div_par_mod(Pe,Ve,fastest_espeed, use_slope_limiter) - (2. / 3) * Pe * Div_par(Ve);
      }
      ddt(Pe) += TE_Pe_parflow;
    } // End Pe_parflow


    if (Pe_conduction){//Row 3
      TRACE("Pe_conduction");
      
      if (!use_new_conduction){
	TE_Pe_conduction = (2.0 / 3.0) * Div_par_K_Grad_par(kappa_epar, Te);
      } else {
	TE_Pe_conduction = (2.0/3.0) * Div_par_K_Grad_par_mod(kappa_epar,Te,false);
      }
      
      if (use_conduction_limiter){
	TE_Pe_conduction = term_limiter(TE_Pe_conduction, conduction_limiter_value);
      }
      
      
      ddt(Pe) += TE_Pe_conduction;
    } // End Pe_conduction
  

    if (Pe_ohmic){//Row 4 Term 3
      TRACE("Pe_ohmic");
      TE_Pe_ohmic = nu * Jpar * (Jpar) / Ne;
      ddt(Pe) += TE_Pe_ohmic;
    } // End Pe_ohmic


    if (Pe_thermalforce){//Row 4 Term 2
      if(!use_new_grad_par){
	TE_Pe_thermalforce = -(2. / 3) * 0.71 * Jpar * Grad_par(Te);
      } else {
	TE_Pe_thermalforce = -(2. / 3) * 0.71 * Jpar * Grad_par_mod(Te);
      }
      ddt(Pe) += TE_Pe_thermalforce;
    } // End Pe_thermalforce


    if (Pe_thermalcurrent){//Row 4 Term 1
      if (!use_new_div_par){
	Field3D tejpar = mul_all(Te,Jpar);
	TE_Pe_thermalcurrent = (2. / 3) * 0.71 * Div_par(tejpar);
      } else {
	TE_Pe_thermalcurrent = (2. / 3) * 0.71 * Div_par_mod(Te,Jpar,fastest_espeed, use_slope_limiter);
      }
      ddt(Pe) += TE_Pe_thermalcurrent;
    } //End Pe_thermalcurrent


    if (Pe_collision){
      throw BoutException("Pe_collision not implemented!");
    } //End Pe_collision


    if (Pe_anomalous){//Row 6
      TRACE("Pe anomalous transport");
      //TE_Pe_anomalous = FCIDiv_a_Grad_perp(mul_all(a_d3d, Te), Ne) + (2. / 3) * FCIDiv_a_Grad_perp(mul_all(a_chi3d, Ne), Te);
      if (use_new_divagradperp){
	TE_Pe_anomalous = (2.0/3.0)*(Div_a_Grad_perp_mod(mul_all(Te,a_d3d), Ne) + Div_a_Grad_perp_mod(mul_all(Ne, a_chi3d), Te));
	
      } else {
	TE_Pe_anomalous = (2.0 / 3.0) * (Div_a_Grad_perp_curv(mul_all(Te,a_d3d), Ne) + Div_a_Grad_perp_curv(mul_all(Ne, a_chi3d), Te));
      }
      ddt(Pe) += TE_Pe_anomalous;
    } // End Pe_anomalous


    if (Pe_energyexchange){//Row 7 Term 3
      TRACE("Pe energy exchange");
      TE_Pe_energyexchange = -(2. / 3) * Wi;
      ddt(Pe) += TE_Pe_energyexchange;
    } // End Pe_energyexchange


    if (Pe_sources){//Row 7 Term 1
      TRACE("Pe sources");
      if (adapt_source){
	TE_Pe_sources = adaptive_sourceterm(Te ,PeSource, Te_target, adaptive_overshoot);
      } else {
	TE_Pe_sources = PeSource;
      }

      if (evolve_neutrals && neutralplasmainteraction){
	TE_Pe_sources += -(2.0/3.0) * Rn; 
      }
      
      ddt(Pe) += TE_Pe_sources;
    } //End Pe_sources


    if (parallel_sheaths){
      switch (par_sheath_model) {
      case 0 :{
	TE_Pe_sheath = sheath_dpe;
	ddt(Pe) += TE_Pe_sheath;
	
	break;
      } 
      } // End switch
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


    if (Pe_dampening){
      TRACE("Electron pressure dampening");
      TE_Pe_dampening = 0.0;
      BOUT_FOR(i, Pe.getRegion("RGN_NOBNDRY")){
        if(Te[i] > Pe_dampening_Te){
          BoutReal tmp = abs(Te[i]) - Pe_dampening_Te;
          TE_Pe_dampening[i] = -Pe_dampening_factor * (floor(exp(tmp),0.0));
        } 
      }
      ddt(Pe) += TE_Pe_dampening ; 
    } // End Pe_dampening
    
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
      if (!use_new_div_par){
	Field3D pivi = mul_all(Pi,Vi);
	TE_Pi_parflow = -Div_par(pivi);
	TE_Pi_parflow += -(2. / 3) * Pi * Div_par(Vi);
      } else if (use_rhie_interpolation){
	TE_Pi_parflow = -Div_par_rhie(Pi,Vi,add_all(Pi,Pe), rhie_cor_up, rhie_cor_down);
	TE_Pi_parflow += -2.0/3.0 * Pi * Div_par_rhie(oness, Vi, add_all(Pi,Pe), rhie_cor_up, rhie_cor_down);
      } else {
	TE_Pi_parflow = -Div_par_mod(Pi,Vi,fastest_ispeed, use_slope_limiter);
	TE_Pi_parflow += -(2. / 3) * Pi * Div_par(Vi);
      }
      ddt(Pi) += TE_Pi_parflow;
    } // End Pi_parflow


    if (Pi_diamagenergyexchange){//Row 3 Term 1 and Term 2
      TRACE("Pi energy exchange with diamag flows");
      if (!use_new_grad_par){
	TE_Pi_diamagenergyexchange = -(2. / 3) * Jpar * Grad_par(Pi);
      } else {
	TE_Pi_diamagenergyexchange = -(2. / 3) * Jpar * Grad_par_mod(Pi);
      }
      TE_Pi_diamagenergyexchange += Pi * fci_curvature(add_all(Pi , Pe),use_bracket);
      ddt(Pi) += TE_Pi_diamagenergyexchange;
    } // End Pi_diamagenergyexchange
    

    if (Pi_conduction){//Row 5 Term 1
      TRACE("Pi thermal conduction");
      if (!use_new_conduction){
	TE_Pi_conduction = (2. / 3) * Div_par_K_Grad_par(kappa_ipar, Ti);
      } else {
	TE_Pi_conduction = (2. / 3) * Div_par_K_Grad_par_mod(kappa_ipar, Ti);
      }
      ddt(Pi) += TE_Pi_conduction;
    } // End Pi_conduction 

    
    if (Pi_resistivedrift){
      throw BoutException("Ion resistive drift not implemented!");
    } // End Pi_resistivedrift
    

    if (Pi_perpviscous){
      throw BoutException("Ion pressure perpendicular viscosity heating not implemented");
    } // End Pi_perpviscous


    if (Pi_sources){//Row 8 Term 1
      if (adapt_source){
	TE_Pi_sources = adaptive_sourceterm(Ti ,PiSource, Ti_target, adaptive_overshoot);
      } else {
	TE_Pi_sources = PiSource;
      }
	
      if (evolve_neutrals && neutralplasmainteraction){
	TE_Pi_sources += -(2.0/3.0) * Qin;
      }
      
      ddt(Pi) += TE_Pi_sources;
    } // End Pi_sources


    if (Pi_anomalous){
      TRACE("Ion anomalous transport");
      if (use_new_divagradperp){
	TE_Pi_anomalous = (2.0/3.0) * (Div_a_Grad_perp_mod(mul_all(Ti,a_d3d), Ne) + Div_a_Grad_perp_mod(mul_all(Ne, a_chi3d), Ti));

      } else {
	TE_Pi_anomalous = (2.0/3.0) * (Div_a_Grad_perp_curv(mul_all(Ti,a_d3d), Ne) + Div_a_Grad_perp_curv(mul_all(Ne, a_chi3d), Ti));
      }
      ddt(Pi) += TE_Pi_anomalous;
    } // End Pi_anomalous


    if (Pi_energyexchange){//Row 8 Term 3
      TE_Pi_energyexchange = (2. / 3) * Wi;
      ddt(Pi) += TE_Pi_energyexchange;
    } // End Pi_energyexchange


    if (parallel_sheaths){
      switch (par_sheath_model) {
      case 0 :{
	ddt(Pi) += sheath_dpi;
	
	break;
      } // End Case 1
      } // End Swith 
    } //End parallel_sheaths

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


  //////////////////////////////////////////////////////////////////////////////////////////////////////////////                                                                                                                                                                 
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////                                                                                                                                                                 
  //                                              Neutral  equations                                          //                                                                                                                                                                 
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////                                                                                                                                                                 
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
  

  if (evolve_neutrals){
    ddt(Nn) = 0.0;
    ddt(NnVn) = 0.0;


  //  TE_Nn_sources = 0.0;

    if (Nn_parflow){
      
      if (!use_new_div_par){
	TE_Nn_parflow = -Div_par(NnVn);
      } else {
	TE_Nn_parflow = -Div_par_mod(Nn,Vn,fastest_ispeed, use_slope_limiter);
      }
      
      ddt(Nn) += TE_Nn_parflow;
    } // End Nn_parflow
    
    
    if (Nn_perpflow){
      if (use_new_divagradperp){
	TE_Nn_perpflow =  Div_a_Grad_perp_mod(div_all(Dnn, Ti),Pn);
      } else {
	//TE_Nn_perpflow = div_all(Dnn, Tn) * new_Delp2(Pn);
	TE_Nn_perpflow = Div_a_Grad_perp_curv(div_all(Dnn, Ti),Pn);
      }      
      ddt(Nn) += TE_Nn_perpflow;
    } // End Nn_perpflow

    
    TE_Nn_sources = 0.0;
    if (Nn_sources && neutralplasmainteraction){
      TE_Nn_sources += Sneutral;
    }
    if (Nn_sources && Recycling_coef > 0.0){
      TE_Nn_sources += Recycling_flux;
    }
    ddt(Nn) += TE_Nn_sources;

    
    if (Nn_hyper){      
      TE_Nn_hyper = hyperdissipation(hyper_D, Nn);
      ddt(Nn) += TE_Nn_hyper;
    } // End Pe_hyper

    
      
    if (NnVn_parflow){      
      if (!use_new_div_par){
	Field3D NnVnVn = mul_all(NnVn, Vn);
	TE_NnVn_parflow = -Div_par(NnVnVn);
      } else {
	TE_NnVn_parflow = -Div_par_mod(NnVn,Vn,fastest_ispeed, use_slope_limiter);
      }
      
      ddt(NnVn) += TE_NnVn_parflow;
    } // End NnVn_parflow 
    
    if (NnVn_perpflow){
      if (use_new_divagradperp){
	TE_NnVn_perpflow =  Div_a_Grad_perp_mod(div_all(mul_all(Vn, Dnn),Ti),Pn);
      } else {
	//TE_NnVn_perpflow = Vn * Dnn / Tn * new_Delp2(Pn); 
	TE_NnVn_perpflow =  Div_a_Grad_perp_curv(div_all(mul_all(Vn, Dnn) ,Ti),Pn);
      }
      ddt(NnVn) += TE_NnVn_perpflow;
    } // End NnVn_perpflow
  
    if (NnVn_pargradient){
      if (!use_new_grad_par){
	TE_NnVn_pargradient = -Grad_par(Pn);
      } else {
	TE_NnVn_pargradient = -Grad_par_mod(Pn);
      }
      ddt(NnVn) += TE_NnVn_pargradient;
    } // End NnVn_pargradient

    if (NnVn_pardiffusion){
      if (!use_new_conduction){
	TE_NnVn_pardiffusion = Div_par_K_Grad_par(mul_all(Dnn, Nn), Vn);
      } else {	
	TE_NnVn_pardiffusion = Div_par_K_Grad_par_mod(mul_all(Dnn, Nn), Vn);
      }
      ddt(NnVn) += TE_NnVn_pardiffusion;
    } // End NnVn_pardiffusion

    if (NnVn_friction && neutralplasmainteraction){
      TE_NnVn_friction = (Vi - Vn) * (Rrc + Rcx);
      ddt(NnVn) += TE_NnVn_friction;
    }

    if (NnVn_hyper){
      TE_NnVn_hyper = hyperdissipation(hyper_nu, NnVn);
      ddt(NnVn) += TE_NnVn_hyper;
    } // End Pe_hyper


    
  } // End evolve_neutrals


  
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
  if (!neutralSolver){
    auto& optss = Options::root();
    neutralSolver = Laplacian::create(&optss["neutralSolver"]);
    neutralSolver->setCoefA(1.0);
  }
  
  neutralSolver->setCoefD(-gamma*Dnn);
  
  ddt(Nn) = neutralSolver->solve(ddt(Nn),oness);
  ddt(NnVn) = neutralSolver->solve(ddt(NnVn),oness);
  return 0;
}




Field3D Hermes::fci_curvature(const Field3D &f, const bool &bool_bracket) {

  // https://www.researchgate.net/publication/328232339_Fluid_simulations_of_plasma_filaments_in_stellarator_geometries_with_BSTING/fulltext/5bdb9d1f92851c6b27a05c8d/Fluid-simulations-of-plasma-filaments-in-stellarator-geometries-with-BSTING.pdf?origin=publication_detail&_tp=eyJjb250ZXh0Ijp7ImZpcnN0UGFnZSI6InByb2ZpbGUiLCJwYWdlIjoicHVibGljYXRpb25Eb3dubG9hZCIsInByZXZpb3VzUGFnZSI6InB1YmxpY2F0aW9uIn19&__cf_chl_tk=__S6xcaYX0LWlh6va5wXxbj0V2x0tPYdBSdKra5LH7Y-1738103131-1.0.1.1-8J2yUk2.guVnqHdn9g7DGjaE5ZbavJsPLJRj._iqTBY
  // Field3D result = mul_all(bracket(logB, f, BRACKET_ARAKAWA), bracket_factor);
  // mesh->communicate(result);
  if (bool_bracket){
    return 2.0 * bracket(logB, f, BRACKET_ARAKAWA) * bracket_factor; // = 
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
  //return a * Grad2_par2(b);
  auto thiscoords=b.getCoordinates();
  return -a * D4DY4(b) / (SQ_all(thiscoords->g_22));
}

Field3D Hermes::term_limiter(const Field3D &a, const BoutReal &val){
  Field3D result{zeroFrom(a)};
  BOUT_FOR(i, a.getRegion("RGN_NOBNDRY")){
    if(a[i] > val){
      result[i] = val;
    } else if (a[i] < (-val)){
      result[i] = -val;
    } else {
      result[i] = a[i];
    }
  }
  return result;
}



Field3D Hermes::Grad_parP(const Field3D &f) {
  return Grad_par(f); //+ 0.5*beta_e*bracket(psi, f, BRACKET_ARAKAWA);
}

Field3D Hermes::Div_parP(const Field3D &f, const bool newop) {
  if (!newop){
    return Div_par(f);
  } else {
    Mesh* mesh = f.getMesh();

    Field3D result{zeroFrom(f)};

    Coordinates* coord = f.getCoordinates();

    BOUT_FOR(i, result.getRegion("RGN_NOBNDRY")) {
      // Calculate flux at upper surface                                                                                                                
      // coord->J.yup()[ind.yp()];                                                                                                                      

      const auto iyp = i.yp();
      const auto iym = i.ym();

      BoutReal c = 0.5 * (f[i] + f.yup()[iyp]);             // K at the upper boundary                                                               
      BoutReal J = 0.5 * (coord->J[i] + coord->J.yup()[iyp]); // Jacobian at boundary                                                                
      BoutReal sqrtg_22 = sqrt(0.5 * (coord->g_22[i] + coord->g_22.yup()[iyp]));
      BoutReal flux = c * J / sqrtg_22;
      result[i] += flux / (coord->dy[i] * coord->J[i]);


      // Calculate flux at lower surface                                                                                                                

      c = 0.5 * (f[i] + f.ydown()[iym]);           // K at the lower boundary                                                                        
      J = 0.5 * (coord->J[i] + coord->J.ydown()[iym]); // Jacobian at boundary                                                                       
      sqrtg_22 = sqrt(0.5 * (coord->g_22[i] + coord->g_22.ydown()[iym]));
      flux = c * J / sqrtg_22;
      result[i] -= flux / (coord->dy[i] * coord->J[i]);

    }
    return result;
  }
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

