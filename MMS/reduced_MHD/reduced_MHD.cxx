
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



const Field3D Div_n_bxGrad_f_B_XPPM(const Field3D &n, const Field3D &f,
                                    bool bndry_flux, bool poloidal,
                                    bool positive) {
  Field3D result{0.0};

  Coordinates *coord = mesh->getCoordinates();
  
  //////////////////////////////////////////
  // X-Z advection.
  //
  //             Z
  //             |
  //
  //    fmp --- vU --- fpp
  //     |      nU      |
  //     |               |
  //    vL nL        nR vR    -> X
  //     |               |
  //     |      nD       |
  //    fmm --- vD --- fpm
  //

  int nz = mesh->LocalNz;
  for (const auto& ind : f.getRegion("RGN_NOBNDRY")) {
    auto kp = ind.zp();
    auto km = ind.zm();

    // 1) Interpolate stream function f onto corners fmp, fpp, fpm

    BoutReal fmm = 0.25 * (f[ind] + f[ind.xm()] + f[km] +
			   f[km.xm()]);
    BoutReal fmp = 0.25 * (f[ind] + f[kp] + f[ind.xm()] +
			   f[kp.xm()]); // 2nd order accurate
    BoutReal fpp = 0.25 * (f[ind] + f[kp] + f[ind.xp()] +
			   f[kp.xp()]);
    BoutReal fpm = 0.25 * (f[ind] + f[ind.xp()] + f[km] +
			   f[km.xp()]);

    // 2) Calculate velocities on cell faces

    BoutReal vU = 0.5 * (coord->J[ind] + coord->J[kp]) * (fmp - fpp) /
      coord->dx[ind]; // -J*df/dx
    BoutReal vD = 0.5 * (coord->J[ind] + coord->J[km]) * (fmm - fpm) /
      coord->dx[ind]; // -J*df/dx

    BoutReal vR = 0.5 * (coord->J[ind] + coord->J[ind.xp()]) * (fpp - fpm) /
      coord->dz[ind]; // J*df/dz
    BoutReal vL = 0.5 * (coord->J[ind] + coord->J[ind.xm()]) * (fmp - fmm) /
      coord->dz[ind]; // J*df/dz

    // output.write("NEW: (%d,%d,%d) : (%e/%e, %e/%e)\n", i,j,k,vL,vR,
    // vU,vD);

    // 3) Calculate n on the cell faces. The sign of the
    //    velocity determines which side is used.

    // X direction
    Stencil1D s;
    s.c = n[ind];
    s.m = n[ind.xm()];
    s.p = n[ind.xp()];
#if CHECK > 1
    s.pp = s.mm = BoutNaN;
#endif

    MC(s);

    // Right side
    if ((mesh->lastX()) && (ind.x() == mesh->xend)) {
      // At right boundary in X

      if (bndry_flux) {
	BoutReal flux;
	if (vR > 0.0) {
	  // Flux to boundary
          flux = vR * s.R ;
        } else {
          // Flux in from boundary
          flux = vR * 0.5 * (n[ind.xp()] + n[ind]) ;
        }
        result[ind] += flux / (coord->dx[ind] * coord->J[ind]);
        result[ind.xp()] -=
	  flux / (coord->dx[ind.xp()] * coord->J[ind.xp()]);
      }
    } else {
      // Not at a boundary
      if (vR > 0.0) {
	// Flux out into next cell
        BoutReal flux = vR * s.R ;
        result[ind] += flux / (coord->dx[ind] * coord->J[ind]);
        result[ind.xp()] -= flux / (coord->dx[ind.xp()] * coord->J[ind.xp()]);
      }
    }

    // Left side

    if ((mesh->firstX()) && (ind.x() == mesh->xstart)) {
      // At left boundary in X

      if (bndry_flux) {
	BoutReal flux;

	if (vL < 0.0) {
	  // Flux to boundary
	  flux = vL * s.L;
	} else {
	  // Flux in from boundary
	  flux = vL * 0.5 * (n[ind.xm()] + n[ind]);
	}

        result[ind] -= flux / (coord->dx[ind] * coord->J[ind]);
        result[ind.xm()] += flux / (coord->dx[ind.xm()] * coord->J[ind.xm()]);
      }
    } else {
      // Not at a boundary

      if (vL < 0.0) {
        const BoutReal flux = vL * s.L ;
        result[ind] -= flux / (coord->dx[ind] * coord->J[ind]);
        result[ind.xm()] += flux / (coord->dx[ind.xm()] * coord->J[ind.xm()]);
      }
    }

    /// NOTE: Need to communicate fluxes

    // Z direction
    s.m = n[km];
    s.p = n[kp];
#if CHECK > 1
    s.pp = s.mm = BoutNaN;
#endif

    // Upwind(s, coord->dz);
    // XPPM(s, coord->dz);
    // Fromm(s, coord->dz);
    MC(s);

    if (vU > 0.0) {
      BoutReal flux = vU * s.R ; 
      result[ind] += flux / (coord->J[ind] * coord->dz[ind]);
      result[kp] -= flux / (coord->J[kp] * coord->dz[kp]);
    }
    if (vD < 0.0) {
      BoutReal flux = vD * s.L ; 
      result[ind] -= flux / (coord->J[ind] * coord->dz[ind]);
      result[km] += flux  / (coord->J[km] * coord->dz[km]);
    }
  }
  FV::communicateFluxes(result);

  //////////////////////////////////////////
  // X-Y advection.
  //
  //
  //  This code does not deal with corners correctly. This may or may not be
  //  important.
  //
  // 1/J d/dx ( J n (g^xx g^yz / B^2) df/dy) - 1/J d/dy( J n (g^xx g^yz / B^2)
  // df/dx )
  //
  // Interpolating stream function f_in onto corners fmm, fmp, fpp, fpm
  // is complicated because the corner point in X-Y is not communicated
  // and at an X-point it is shared with 8 cells, rather than 4
  // (being at the X-point itself)
  // Corners also need to be shifted to the correct toroidal angle
  ASSERT1(! poloidal)
  
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
  Field3D phi_boundary;
  bool U_ExB,U_Delp2,U_gradpar;
  Field3D bracket_factor;

  
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

    OPTION(optMHD,U_ExB,false);
    OPTION(optMHD,U_Delp2,false);
    OPTION(optMHD,U_gradpar,false);
    
    bracket_factor = sqrt(coord->g_22) / (coord->J);
    
    xl = opt["xl"].withDefault(Field3D{0.0});
    yl = opt["yl"].withDefault(Field3D{0.0});
    zl = opt["zl"].withDefault(Field3D{0.0});
    SAVE_ONCE(xl,yl,zl);
    phi_boundary = 0.0;
    SAVE_REPEAT(phi_boundary);
    TRACE("SET VARIABLES");
    U = 0.0;
    Apar = 0.0;
    Jpar = 0.0;
    phi = 0.0;
    mesh->communicate(Apar,Jpar,phi,U);    
    SOLVE_FOR( U , Apar );
    SAVE_REPEAT( Jpar , phi ,phi_solution);


    TRACE("SET PHI SOLVER");
    phiSolver = Laplacian::create(&opt["phiSolver"]);
    
    return 0;
  }
  
  int rhs(BoutReal t) override {

    
    phi_solution = 0.2*cos(0.8 - 2*yl)*sin(0.3 - 0.2*t)*sin(31.41592653589794*(-0.4 + xl))*sin(0. - 8*zl);
    mesh->communicate(U,Apar,phi_solution);


    // SET BOUNDARIES FOR POTENTIAL AT THE CELL FACES
    phi_boundary = phi_solution;
    
    if (mesh->firstX()) {
      for (int j = mesh->ystart; j <= mesh->yend; j++) {
	for (int k = 0; k < mesh->LocalNz; k++) {
	  phi_boundary(mesh->xstart - 1, j, k) = 0.5*(phi_solution(mesh->xstart - 1, j, k) + phi_solution(mesh->xstart, j, k));	  
	}
      }
    }

    if (mesh->lastX()) {
      for (int j = mesh->ystart; j <= mesh->yend; j++) {
	for (int k = 0; k < mesh->LocalNz; k++) {
 	  phi_boundary(mesh->xend + 1, j, k) = 0.5*(phi_solution(mesh->xend + 1, j, k) + phi_solution(mesh->xend, j, k));

	}
      }
    }

    TRACE("CALCULATE POTENTIAL");
    phi = phiSolver->solve(U,phi_boundary);
    mesh->communicate(phi);


    TRACE("Calculate parallel current");
    
    Jpar = -new_Delp2(Apar);
    mesh->communicate(Jpar);
    
    
    ddt(U) = 0.0;

    TRACE("U time evolution");
    if (evolve_U){
      if (U_gradpar){
	ddt(U) += Div_par(Jpar);
      }
      if (U_ExB){
	ddt(U) +=  -bracket(phi_solution,U,BRACKET_ARAKAWA)*bracket_factor;
	//ddt(U) -= Div_n_bxGrad_f_B_XPPM(U, phi_solution, true, false,false);
      }
      
      if (U_Delp2){
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
