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
using bout::globals::mesh;



BoutReal BOUTMIN(const BoutReal &a, const BoutReal &b, const BoutReal &c,
                 const BoutReal &d) {
  BoutReal r1 = (a < b) ? a : b;
  BoutReal r2 = (c < d) ? c : d;
  return (r1 < r2) ? r1 : r2;
}

struct Stencil1D {
  // Cell centre values
  BoutReal c, m, p, mm, pp;

  // Left and right cell face values
  BoutReal L, R;
};

// First order upwind for testing
void Upwind(Stencil1D &n) { n.L = n.R = n.c; }

// Fromm method
void Fromm(Stencil1D &n) {
  n.L = n.c - 0.25 * (n.p - n.m);
  n.R = n.c + 0.25 * (n.p - n.m);
}





Field3D Div_a_Grad_perp_mod(const Field3D& b, const Field3D& a){
  auto *coord = mesh->getCoordinates();
  Field3D tmp = (DDX(coord->J * coord->g11*b)*DDX(a) + coord->J * coord->g11 * b * D2DX2(a))/coord->J;
  tmp += (DDZ(coord->J * coord->g33 * b)*DDZ(a) + coord->J * coord->g33 * b * D2DZ2(a))/coord->J;
  tmp += (DDX(coord->J * coord->g13 * b)*DDZ(a) + coord->J * coord->g13 * b * D2DXDZ(a) * 2.0 + DDZ(coord->J * coord->g13 * b)*DDX(a))/coord->J;
  return tmp;

}




/// The minmod function returns the value with the minimum magnitude
/// If the inputs have different signs then returns zero
BoutReal minmod(BoutReal a, BoutReal b) {
  if (a * b <= 0.0)
    return 0.0;

  if (fabs(a) < fabs(b))
    return a;
  return b;
}

BoutReal minmod(BoutReal a, BoutReal b, BoutReal c) {
  // If any of the signs are different, return zero gradient
  if ((a * b <= 0.0) || (a * c <= 0.0)) {
    return 0.0;
  }

  // Return the minimum absolute value
  return SIGN(a) * BOUTMIN(fabs(a), fabs(b), fabs(c));
}

void MinMod(Stencil1D &n) {
  // Choose the gradient within the cell
  // as the minimum (smoothest) solution
  BoutReal slope = minmod(n.p - n.c, n.c - n.m);
  n.L = n.c - 0.5 * slope; // 0.25*(n.p - n.m);
  n.R = n.c + 0.5 * slope; // 0.25*(n.p - n.m);
}

// Monotonized Central limiter (Van-Leer)
void MC(Stencil1D &n) {
  BoutReal slope =
      minmod(2. * (n.p - n.c), 0.5 * (n.p - n.m), 2. * (n.c - n.m));
  n.L = n.c - 0.5 * slope;
  n.R = n.c + 0.5 * slope;
}

const Field3D Div_par_K_Grad_par_mod(const Field3D& K, const Field3D& f, bool bndry_flux) {
  TRACE("FV::Div_par_K_Grad_par_mod");

  TRACE("Check first field");
  ASSERT2(K.getLocation() == f.getLocation());

  ASSERT2(K.hasParallelSlices());

  TRACE("Check second field");
  ASSERT2(f.hasParallelSlices());

  
  Mesh* mesh = K.getMesh();

  Field3D result{zeroFrom(f)};

  Coordinates* coord = f.getCoordinates();

  BOUT_FOR(i, result.getRegion("RGN_NOBNDRY")) {
    // Calculate flux at upper surface
    // coord->J.yup()[ind.yp()];
    
    const auto iyp = i.yp();
    const auto iym = i.ym();

    BoutReal c = 0.5 * (K[i] + K.yup()[iyp]);             // K at the upper boundary                                                                
    BoutReal J = 0.5 * (coord->J[i] + coord->J.yup()[iyp]); // Jacobian at boundary                                                                  
    BoutReal g_22 = 0.5 * (coord->g_22[i] + coord->g_22.yup()[iyp]);                                                                                 
    BoutReal gradient = 2. * (f.yup()[iyp] - f[i]) / (coord->dy[i] + coord->dy[i]);                                                          
    BoutReal flux = c * J * gradient / g_22;                                                                                                         
    result[i] += flux / (coord->dy[i] * coord->J[i]);                                                                                                
                                                                                                                                                     
                                                                                                                                                  
    // Calculate flux at lower surface                                                                                                               
                                                                                                                                                    
    c = 0.5 * (K[i] + K.ydown()[iym]);           // K at the lower boundary                                                                          
    J = 0.5 * (coord->J[i] + coord->J.ydown()[iym]); // Jacobian at boundary                                                                         
    g_22 = 0.5 * (coord->g_22[i] + coord->g_22.ydown()[iym]);                                                                                        
    gradient = 2. * (f[i] - f.ydown()[iym]) / (coord->dy[i] + coord->dy[i]);                                                               
    flux = c * J * gradient / g_22;                                                                                                                  
    result[i] -= flux / (coord->dy[i] * coord->J[i]);
    
  }


  return result;
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


const Field3D new_Delp2(const Field3D& a){
  auto *coord = mesh->getCoordinates();
  Field3D tmp = (DDX(coord->J * coord->g11)*DDX(a) + coord->J * coord->g11 * D2DX2(a))/coord->J;
  tmp += (DDZ(coord->J * coord->g33)*DDZ(a) + coord->J * coord->g33 * D2DZ2(a))/coord->J;
  tmp += (DDX(coord->J * coord->g13)*DDZ(a) + coord->J * coord->g13 * D2DXDZ(a) *2.0 + DDZ(coord->J * coord->g13)*DDX(a))/coord->J;
  return tmp;
}

const Field3D power(const Field3D& a, const int powval){
  return pow(a,powval);
}


class distorted_grid : public PhysicsModel {
private:
    Field3D Ne;	
  Field3D theta,rho_n,rho;
  BoutReal diffusion,dpar;
  Field3D diffusion3D, dpar3D;
  Field3D Ne_source,Ne_solution,xl,yl,zl, x1;
  Field3D debug_diffusion, debug_pardiffusion, debug_arakawa;
  Field3D phi_solution;
  Field3D bracket_factor;

  Field3D delp2_Ne;
  Field3D inverted_Ne;
  Field3D forward_Ne, forward_Ne_solution;
  
  std::unique_ptr<Laplacian> phiSolver{nullptr};
  Field3D phiSolverbndry;
protected:
  int init(bool UNUSED(restart)) override {

    TRACE("LOAD DATA AND OPTIONS");
    auto& opt = Options::root();
    auto *coord = mesh->getCoordinates();
    auto& optne = Options::root()["Ne"];

    OPTION(opt,diffusion,0.0);
    OPTION(opt,dpar,0.0);
    xl = opt["xl"].withDefault(Field3D{0.0});
    yl = opt["yl"].withDefault(Field3D{0.0});
    zl = opt["zl"].withDefault(Field3D{0.0});
    x1 = opt["x1"].withDefault(Field3D{0.0});
    SAVE_ONCE(xl,yl,zl,x1);    

    Ne=0.0;
    
    SOLVE_FOR(Ne);

    Ne_source = 0.0;
    Ne_solution = 0.0;
    phi_solution = 0.0;
    SAVE_REPEAT(Ne_source, Ne_solution, phi_solution);
    
    theta=0.0;
    mesh->get(theta,"theta");
    SAVE_ONCE(theta);

    rho_n=0.0;
    mesh->get(rho_n,"rho_n");
    SAVE_ONCE(rho_n);

    rho=0.0;
    mesh->get(rho,"rho");
    SAVE_ONCE(rho);


    xl=rho;
    zl=theta;

    bracket_factor = sqrt(coord->g_22) / (coord->J);

    debug_diffusion = 0.0;
    debug_pardiffusion = 0.0;
    debug_arakawa = 0.0;
    SAVE_REPEAT(debug_diffusion,debug_pardiffusion, debug_arakawa);

    diffusion3D = opt["diffusion"].withDefault(Field3D{0.0});
    dpar3D = opt["dpar"].withDefault(Field3D{0.0});

    diffusion3D = 0.001 + 0.001*xl;
    dpar3D = 0.3 + 0.02 * sin(4 * zl + 0.5);
    
    SAVE_ONCE(diffusion3D, dpar3D);

    phiSolver = Laplacian::create(&opt["phiSolver"]);

    phiSolver->setCoefD(Field3D{1.0});
    
    inverted_Ne = 0.0;
    phiSolverbndry = 0.0;
    delp2_Ne = 0.0;
    forward_Ne = 0.0;
    forward_Ne_solution = 0.0;
    SAVE_REPEAT(inverted_Ne, phiSolverbndry, delp2_Ne, forward_Ne, forward_Ne_solution);
    return 0;
  }
  
  int rhs(BoutReal t) override {
    
    //auto *coord = mesh->getCoordinates();

    Ne_solution = 0.1*cos(0.5 - 1.*yl)*sin(0. - 0.01*t)*sin(15.70796326794897*(-0.4 + xl))*sin(0.1 - 4*zl);

    Ne_source = -0.0015707963267948969*cos(15.70796326794897*(-0.4 + xl))*cos(0.5 - 1.*yl)*sin(0. - 0.01*t)*sin(0.1 - 4*zl) - (1.5707963267948968*(0.001 + 0.001*xl)*cos(15.70796326794897*(-0.4 + xl))*cos(0.5 - 1.*yl)*sin(0. - 0.01*t)*sin(0.1 - 4*zl))/xl - 0.001*cos(0. - 0.01*t)*cos(0.5 - 1.*yl)*sin(15.70796326794897*(-0.4 + xl))*sin(0.1 - 4*zl) + 24.674011002723407*(0.001 + 0.001*xl)*cos(0.5 - 1.*yl)*sin(0. - 0.01*t)*sin(15.70796326794897*(-0.4 + xl))*sin(0.1 - 4*zl) - (1.*(0. - 1.6*(0.001 + 0.001*xl)*cos(0.5 - 1.*yl)*sin(0. - 0.01*t)*sin(15.70796326794897*(-0.4 + xl))*sin(0.1 - 4*zl)))/power(xl,2) + 0.01*((-3.1415926535897936*cos(15.70796326794897*(-0.4 + xl))*cos(0.5 - 1.*yl)*cos(0.1 - 4*zl)*cos(1.*zl)*power(sin(0. - 0.01*t),2)*sin(15.70796326794897*(-0.4 + xl))*sin(6.*yl))/xl + (0.7853981633974484*cos(15.70796326794897*(-0.4 + xl))*cos(0.5 - 1.*yl)*power(sin(0. - 0.01*t),2)*sin(15.70796326794897*(-0.4 + xl))*sin(6.*yl)*sin(0.1 - 4*zl)*sin(1.*zl))/xl) - (((0.08*cos(0.5 + 4*zl)*((-0.4*cos(0.5 - 1.*yl)*cos(0.1 - 4*zl)*sin(0. - 0.01*t)*sin(15.70796326794897*(-0.4 + xl)))/(8. - 2.*power(xl,2)) + 0.1*sin(0. - 0.01*t)*sin(15.70796326794897*(-0.4 + xl))*sin(0.5 - 1.*yl)*sin(0.1 - 4*zl)))/sqrt(1 + power(xl,2)/power(8. - 2.*power(xl,2),2)) + ((-0.4*cos(0.1 - 4*zl)*sin(0. - 0.01*t)*sin(15.70796326794897*(-0.4 + xl))*sin(0.5 - 1.*yl) - (1.6*cos(0.5 - 1.*yl)*sin(0. - 0.01*t)*sin(15.70796326794897*(-0.4 + xl))*sin(0.1 - 4*zl))/(8. - 2.*power(xl,2)))*(0.3 + 0.02*sin(0.5 + 4*zl)))/sqrt(1 + power(xl,2)/power(8. - 2.*power(xl,2),2)))/(8. - 2.*power(xl,2)) + (((-0.4*cos(0.1 - 4*zl)*sin(0. - 0.01*t)*sin(15.70796326794897*(-0.4 + xl))*sin(0.5 - 1.*yl))/(8. - 2.*power(xl,2)) - 0.1*cos(0.5 - 1.*yl)*sin(0. - 0.01*t)*sin(15.70796326794897*(-0.4 + xl))*sin(0.1 - 4*zl))*(0.3 + 0.02*sin(0.5 + 4*zl)))/sqrt(1 + power(xl,2)/power(8. - 2.*power(xl,2),2)))/sqrt(1 + power(xl,2)/power(8. - 2.*power(xl,2),2));
    
    
    phi_solution = -0.5*cos(1.*zl)*sin(0. - 0.01*t)*sin(15.70796326794897*(-0.4 + xl))*sin(6.*yl);
    mesh->communicate(phi_solution);


    forward_Ne_solution = (1.5707963267948968*cos(15.70796326794897*(-0.4 + xl))*cos(0.5 - 1.*yl)*sin(0. - 0.01*t)*sin(0.1 - 4*zl))/xl - 24.674011002723407*cos(0.5 - 1.*yl)*sin(0. - 0.01*t)*sin(15.70796326794897*(-0.4 + xl))*sin(0.1 - 4*zl) - (1.6*cos(0.5 - 1.*yl)*sin(0. - 0.01*t)*sin(15.70796326794897*(-0.4 + xl))*sin(0.1 - 4*zl))/power(xl,2);

    
    
    Ne.applyBoundary();
      
    if (mesh->firstX()) {
      for (int j = mesh->ystart; j <= mesh->yend; j++) {
	for (int k = 0; k < mesh->LocalNz; k++) {
	  BoutReal bndryval = 0.5*(Ne_solution(1,j,k)+Ne(2,j,k));
	  Ne(1,j,k) = 2.0*bndryval - Ne(2,j,k);
	  BoutReal nextval = 0.5*(Ne_solution(0,j,k)+Ne(1,j,k));
	  Ne(0,j,k) = 2.0*nextval - Ne(1,j,k);
	  //Ne(0,j,k) = Ne_solution(0,j,k);
	  //Ne(1,j,k) =	Ne_solution(1,j,k);
	}
      }
    }

      
    if (mesh->lastX()) {
      int n = mesh->LocalNx;
      for (int j = mesh->ystart; j <= mesh->yend; j++) {
	  for (int k = 0; k < mesh->LocalNz; k++) {
	    BoutReal bndryval = 0.5*(Ne_solution(n-2,j,k)+Ne(n-3,j,k));	
	    Ne(n-2,j,k) = 2.0*bndryval - Ne(n-3,j,k);
	    BoutReal nextval = 0.5*(Ne_solution(n-1,j,k)+Ne(n-2,j,k));
	    Ne(n-1,j,k) = 2.0*nextval - Ne(n-2,j,k);
	    
	    //Ne(n-1,j,k) = Ne_solution(n-1,j,k);
	    //Ne(n-2,j,k) = Ne_solution(n-2,j,k)
	  }
      }
    }



    
    mesh->communicate(Ne);
    Ne.applyParallelBoundary();

    ddt(Ne) = 0.0;
    ddt(Ne) += Ne_source;
    
    debug_diffusion = Div_a_Grad_perp_mod(diffusion3D, Ne);
    ddt(Ne) += debug_diffusion;

    //debug_pardiffusion = dpar3D * Grad2_par2(Ne);
    debug_pardiffusion = Div_par_K_Grad_par_mod(dpar3D, Ne, false);
    ddt(Ne) += debug_pardiffusion;
    
    debug_arakawa = -0.01 * bracket(phi_solution, Ne, BRACKET_ARAKAWA) * bracket_factor;
    ddt(Ne) += debug_arakawa;

    delp2_Ne = new_Delp2(Ne);
    forward_Ne = phiSolver->forward(Ne);

    if (mesh->lastX()) {
      for (int j = mesh->ystart; j <= mesh->yend; j++) {
	for (int k = 0; k < mesh->LocalNz; k++) {
	  phiSolverbndry(mesh->xend + 1, j, k) = 0.5 * ( Ne_solution(mesh->xend + 1, j, k) + Ne_solution(mesh->xend, j, k) ) ;	    
	}
      }
    }
    
    if (mesh->firstX()) {	 
      for (int j = mesh->ystart; j <= mesh->yend; j++) {
	for (int k = 0; k < mesh->LocalNz; k++) {
	  phiSolverbndry(mesh->xstart - 1, j, k) = 0.5 * ( Ne_solution(mesh->xstart, j, k) + Ne_solution(mesh->xstart - 1, j, k) );
	}
      }
    }

    inverted_Ne = phiSolver->solve(delp2_Ne, phiSolverbndry);
    
    
    
    return 0;
  }


};
  
BOUTMAIN(distorted_grid);
