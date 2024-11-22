//#include <bout/derivs.hxx>
#include "Loki.hxx"
#include <bout/physicsmodel.hxx>
#include <field_factory.hxx>
#include <bout/derivs.hxx>
#include "parallel_boundary_region.hxx"
#include "boundary_region.hxx"
#include "../../div_ops.hxx"
#include "../../loadmetric.hxx"
#include <algorithm> // For std::max
#include <initializer_list>

#include <bout/constants.hxx>
#include <bout/assert.hxx>
#include <bout/fv_ops.hxx>
#include <cmath>



template <typename T, typename... Args>
T calculateMax(T first, Args... args) {
    return std::max({first, static_cast<T>(args)...});
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

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


int Loki::init(bool restarting) {
  
  auto& opt = Options::root();
  auto& optNe = opt["Ne"];
  auto& optNVi = opt["NVi"];
  auto& optPe = opt["Pe"];
  auto& optPi = opt["Pi"];
  auto& optVePsi = opt["VePsi"];
  auto& optVort = opt["Vort"];  

  //Support variable initialisation
  x_val= opt["x_val"].withDefault(Field3D{0.0});
  xl = opt["xl"].withDefault(Field3D{0.0});
  yl = opt["yl"].withDefault(Field3D{0.0});
  zl = opt["zl"].withDefault(Field3D{0.0});
  SAVE_ONCE(xl,yl,zl,x_val);

  auto *coord = mesh->getCoordinates();
  g_22 = coord->g_22;


  Mesh* mesh=Ne.getMesh();
  RR=-1.0;
  ZZ=-1.0;
  mesh->get(RR, "R");
  mesh->get(ZZ, "Z");

  theta = 0.0;
  rho = 0.0;
  BOUT_FOR(i, Ne.getMesh()->getRegion3D("RGN_ALL")) {
    //theta[i] = atan(Z[i]/R[i]);
    theta[i] = atan2(ZZ[i],RR[i]);
    if (theta[i] < 0.0){
      theta[i] += 2.0*3.1415926535897935;
    }
    rho[i] = sqrt(ZZ[i]*ZZ[i]+RR[i]*RR[i]);
  }
  

  SAVE_ONCE(RR,ZZ,theta,rho);
  
  OPTION(opt, upwind, false);
  
  Ne_solution = 0.0;
  Ne_source = 0.0;
  Ne_bndry = 0.0;
  SAVE_REPEAT(Ne_solution, Ne_source,Ne_bndry);

  
  ////////////////////////////////////////////////
  
  OPTION(opt, evolve_Ne, false);
  OPTION(opt, evolve_NVi, false);
  OPTION(opt, evolve_Pe, false);
  OPTION(opt, evolve_Pi, false);
  OPTION(opt, evolve_VePsi, false);
  OPTION(opt, Vort, false);
    
  OPTION(optNe, Ne_ExB, false);
  OPTION(optNe, Ne_diamagnetic, false);
  OPTION(optNe, Ne_vpar, false);
  OPTION(optNe, Ne_collisional, false);
  OPTION(optNe, Ne_diffusion_perp, false);
  OPTION(optNe, Ne_diffusion_par, false);
  OPTION(optNe, Ne_sources, false);
  OPTION(optNe, Ne_gradpar, false);
  
  if(evolve_Ne){
    SOLVE_FOR(Ne);
    EvolvingVars.add(Ne);
    alloc_all(Ne);
    SAVE_REPEAT(ddt(Ne));
    D_perp = optNe["D_perp"].withDefault(Field3D{0.0});
    D_par = optNe["D_par"].withDefault(Field3D{0.0});
  }

  if(evolve_NVi){
    SOLVE_FOR(NVi);
    EvolvingVars.add(NVi);
    alloc_all(NVi);
  }

  if(evolve_Pe){
    SOLVE_FOR(Pe);
    EvolvingVars.add(Pe);
    alloc_all(Pe);
  } else {
    Te = optPe["Te"].withDefault(Field3D{0.0});
  }

  if(evolve_Pi){
    SOLVE_FOR(Pi);
    EvolvingVars.add(Pi);
    alloc_all(Pi);
  } else {
    Ti = optPi["Ti"].withDefault(Field3D{0.0});
  }


  if(evolve_VePsi){
    SOLVE_FOR(VePsi);
    EvolvingVars.add(VePsi);
    alloc_all(VePsi);
  } else{
    Ve = optPi["Ve"].withDefault(Field3D{0.0});
    Apar = optPi["Apar"].withDefault(Field3D{0.0});
  }

  if(evolve_Vort){
      SOLVE_FOR(Vort);
      EvolvingVars.add(Vort);
      alloc_all(Vort);
  } else {
    phi = optPi["phi"].withDefault(Field3D{0.0});
  }
  
  return 0;
}

int Loki::rhs(BoutReal t) {

  
  // Calculate the density solution

 
  Ne_solution = 2*cos(0.5 - yl)*sin(0.3 - 0.1*t)*sin(15.70796326794897*(-0.4 + xl))*sin(0.1 - 4*zl);

    
  mesh->communicate(Ne_solution);

  Ne_source = 0.0;
  
  mesh->communicate(Ne);

  
  ////////////////////////////////////////////////////////////////////////////////////////////////////
  //                                   Ne time evolution                                            //  
  ////////////////////////////////////////////////////////////////////////////////////////////////////

  //    Ne_ExB, Ne_mag, Ne_vpar, Ne_collisional ,Ne_diffusion , Ne_sources;
  ddt(Ne) = 0.0;
  if (evolve_Ne){

    if (Ne_gradpar){
      TRACE("Density parallel gradient");
      if(!upwind){
	ddt(Ne) += Grad_par(Ne);
      } else {
	for(auto &i : Ne.getRegion("RGN_NOBNDRY")) {                                                                                                                                                                                                                            
        ddt(Ne)[i] +=  (Ne.ydown()[i.ym()] - Ne[i])/sqrt(g_22[i]);                                                                                                                                                                                                               
	}
      }
    }
    
    if (Ne_vpar){
      TRACE("Density parallel velocity");
      Field3D neve = mul_all(Ne,Ve);
      ddt(Ne) += Div_par(neve);
    }

    if (Ne_diffusion_perp){
      TRACE("Density perpendicular diffusion");
      //ddt(Ne) += FCIDiv_a_Grad_perp(D_perp,Ne);
    }
    
    if (Ne_diffusion_par){
      TRACE("Density parallel diffusion");
      //ddt(Ne) += Div_par_K_Grad_par(D_par,Ne);
      ddt(Ne) += Grad2_par2(Ne);
    }

      
      
  }

    
    
  return 0;
}


