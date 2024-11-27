
#include <bout/physicsmodel.hxx>

#include <bout/constants.hxx>
#include <bout/initialprofiles.hxx>
#include <bout/interpolation.hxx>
#include <bout/invert_laplace.hxx>
#include <bout/invert_parderiv.hxx>
#include <field_factory.hxx>
#include <bout/derivs.hxx>
#include <bout/assert.hxx>
#include <bout/fv_ops.hxx>
#include <cmath>





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







///
class reduced_MHD : public PhysicsModel {
private:
  Field3D U, Apar;
  Field3D Jpar,phi;
  Field3D Bxy;
  bool evolve_U,evolve_Apar;
  BoutReal mu,beta_hat,eta;
  std::unique_ptr<Laplacian> phiSolver{nullptr};
  Field3D phi_solution;
  Field3D xl,yl,zl;
  
  
protected:
  int init(bool UNUSED(restart)) override {

    TRACE("LOAD DATA AND OPTIONS");
    auto& opt = Options::root();
    auto& optMHD = Options::root()["reduced_MHD"];
    auto *coord = mesh->getCoordinates();
    mesh->get(Bxy, "Bxy");
    mesh->communicate(Bxy);

    OPTION(optMHD , mu , 0.0);
    OPTION(optMHD , eta , 0.0);
    OPTION(optMHD,evolve_U,false);
    OPTION(optMHD,evolve_Apar,false);

    xl = opt["xl"].withDefault(Field3D{0.0});
    yl = opt["yl"].withDefault(Field3D{0.0});
    zl = opt["zl"].withDefault(Field3D{0.0});
    SAVE_ONCE(xl,yl,zl);

    
    TRACE("SET VARIABLES");
    U = 0.0;
    Apar = 0.0;
    Jpar = 0.0;
    phi = 0.0;
    mesh->communicate(Apar,Jpar,phi,U);    
    SOLVE_FOR( U , Apar );
    SAVE_REPEAT( Jpar , phi );


    TRACE("SET PHI SOLVER");
    phiSolver = Laplacian::create(&opt["phiSolver"]);
    
    return 0;
  }
  
  int rhs(BoutReal UNUSED(time)) override {

    phi_solution = 0.75*cos(0.8 - 2*yl)*sin(0.3 - 0.5*t)*sin(15.70796326794897*(-0.4 + xl))*sin(0.3 - 8*zl);
    
    mesh->communicate(U,Apar,phi_solution);


    TRACE("CALCULATE POTENTIAL");
    phi = phiSolver->solve(Bxy*U,phi);
    

    
    ddt(U) = 0.0;

    TRACE("U time evolution");
    if (evolve_U){

      ddt(U) += SQ(Bxy) * Grad_par(div_all(Jpar,Bxy));

      ddt(U) -= bracket(phi,U,BRACKET_ARAKAWA);

      if (mu>0.0){
	ddt(U) += mu * new_Delp2(U);
      }

      
    } // evolve_U


    TRACE("Apar time evolution");
    ddt(Apar) = 0.0;
    
    if (evolve_Apar){

      ddt(Apar) -= Grad_par(phi)/beta_hat;

      if (eta>0.0){
	ddt(Apar) -= eta * Jpar / beta_hat;
      }
      
    } // evolve_Apar
    
    
    
    return 0;
  }

  const Field3D new_Delp2(const Field3D& a){
    auto *coord = mesh->getCoordinates();
    Field3D tmp = (DDX(coord->J * coord->g11)*DDX(a) + coord->J * coord->g11 * D2DX2(a))/coord->J;
    tmp += (DDZ(coord->J * coord->g33)*DDZ(a) + coord->J * coord->g33 * D2DZ2(a))/coord->J;
    return tmp;
  }

};
  
BOUTMAIN(reduced_MHD);
