//#include <bout/derivs.hxx>

#include <bout/physicsmodel.hxx>
#include <field_factory.hxx>
#include <bout/derivs.hxx>
#include "parallel_boundary_region.hxx"
#include "boundary_region.hxx"
#include "../../div_ops.hxx"
#include <algorithm> // For std::max
#include <initializer_list>


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

Field3D Div_parP_f(const Field3D &f, const Field3D &v,
                           Field3D &sound_speed) {
 throw BoutException("NI");
}

Field3D Div_parP_n(const Field3D &n, const Field3D &v,
                           Field3D &sound_speed) {
  auto parbc = "parallel_neumann_o2";
  Field3D gn = MinMod(n);
  Field3D gv = MinMod(v);
  n.getMesh()->communicate(gn, gv, sound_speed);
  gn.applyParallelBoundary(parbc);
  gv.applyParallelBoundary(parbc);
  sound_speed.applyParallelBoundary(parbc);
  Field3D result{0.0};

  auto coord = n.getCoordinates();
  BOUT_FOR(i, n.getRegion("RGN_ALL")) {
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
    const BoutReal amaxp = calculateMax(
        abs(v[i]), abs(v.yup()[ip]), sound_speed[i], sound_speed.yup()[ip]);
    const BoutReal amaxm = calculateMax(abs(v[i]), abs(v.ydown()[im]),
                                     sound_speed[i], sound_speed.ydown()[im]);
    BoutReal Gnvp = 0.5 * (niR * SQ(viR) + npL * SQ(vpL)) +
                    0.5 * amaxp * (niR * viR - npL * vpL);
    BoutReal Gnvm = 0.5 * (nmR * SQ(vmR) + niL * SQ(viL)) +
                    0.5 * amaxm * (nmR * vmR - niL * viL);
    ASSERT1(std::isfinite(Gnvp));
    ASSERT1(std::isfinite(Gnvm));
    result[i] = AoVi * (Gnvp - Gnvm);
  }
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


class fluid : public PhysicsModel {
private:
  Field3D n,p,nv;
  Field3D n_solution,p_solution,nv_solution;
  BoutReal gamma;
  Field3D xl,yl,zl;
  Field3D bndry_n,bndry_p,bndry_nv;
  BoutReal Dy;
  Field3D Diss;
  bool dissipation;
protected:
  int init(bool restarting) override {
    auto& opt = Options::root();
    xl=opt["xl"].withDefault(Field3D{0.0});
    yl=opt["yl"].withDefault(Field3D{0.0});
    zl=opt["zl"].withDefault(Field3D{0.0});
    
    gamma = opt["gamma"].doc("Adiabatic index (ratio of specific heats)").withDefault(5. / 3);
    Diss = 0.0;
    Dy = 0.0;
    Dy=opt["Dy"].withDefault(0.0);
    dissipation = opt["dissipation"].withDefault<bool>(false);
    Diss = 1.0 * Dy;
    
    Diss.applyBoundary("neumann_o2");
    mesh->communicate(Diss);
    Diss.applyParallelBoundary("parallel_neumann_o2");
    
    n_solution=0.0;
    p_solution = 0.0;
    nv_solution = 0.0;

    bndry_n = 0.0;
    bndry_p = 0.0;
    bndry_nv = 0.0;
    
    SAVE_REPEAT(n_solution,p_solution,nv_solution,bndry_n,bndry_p,bndry_nv);
    
    SOLVE_FOR(n, p, nv);
    return 0;
  }

  int rhs(BoutReal t) override {
    n.applyBoundary("neumann_o2");
    p.applyBoundary("neumann_o2");
    nv.applyBoundary("neumann_o2");
    
    // Calculate the solution of N
    
    n_solution= -0.1 * sin(t - 2.0*yl) + 1;
    
    p_solution = 0.1*cos(t + 3.0*yl) + 1;

    
    nv_solution = 0.1 * sin(2.0 * t + yl);

    n_solution.applyBoundary("neumann_o2");
    p_solution.applyBoundary("neumann_o2");
    nv_solution.applyBoundary("neumann_o2");
    mesh->communicate(n_solution,p_solution,nv_solution);
    //n_solution.applyParallelBoundary("parallel_neumann_o2");
    //p_solution.applyParallelBoundary("parallel_neumann_o2");
    //nv_solution.applyParallelBoundary("parallel_neumann_o2");
    // Apply parallel boundary conditions by hand
    /*
    for (const auto &bndry_par :
	   mesh->getBoundariesPar()) {
      for (const auto &pnt : *bndry_par) {
	int xx = pnt.ind().x();
	int yy = pnt.ind().y();
	int zz = pnt.ind().z();
	n(xx,yy,zz) = n_solution(xx,yy,zz);
	p(xx,yy,zz) = p_solution(xx,yy,zz);
	nv(xx,yy,zz) = nv_solution(xx,yy,zz);
      }
    }
    */
    mesh->communicate(n,p,nv);
    //n.applyParallelBoundary("parallel_neumann_o2");
    //p.applyParallelBoundary("parallel_neumann_o2");
    //nv.applyParallelBoundary("parallel_neumann_o2");

    for (const auto &bndry_par :
	   mesh->getBoundariesPar()) {
      for (const auto &pnt : *bndry_par) {
	int xx = pnt.ind().x();
	int yy = pnt.ind().y();
	int zz = pnt.ind().z();
	BoutReal N_parvalue = 0.0;
	if (bndry_par->dir > 0.0){
	  bndry_n(xx,yy,zz) = (n_solution(xx,yy+1,zz));
	  bndry_p(xx,yy,zz) = (p_solution(xx,yy+1,zz));
	  bndry_nv(xx,yy,zz) = (nv_solution(xx,yy+1,zz));
	} else {
	  //bndry_N(xx,yy,zz) = (N_solution(xx,yy-1,zz)+N_solution(xx,yy,zz))/2.0;
	  bndry_n(xx,yy,zz) = (n_solution(xx,yy-1,zz));
	  bndry_p(xx,yy,zz) = (p_solution(xx,yy-1,zz));
	  bndry_nv(xx,yy,zz) = (nv_solution(xx,yy-1,zz));
	}
	n.ynext(bndry_par->dir)(xx,yy+bndry_par->dir,zz) = bndry_n(xx,yy,zz);
	p.ynext(bndry_par->dir)(xx,yy+bndry_par->dir,zz) = bndry_p(xx,yy,zz);
	nv.ynext(bndry_par->dir)(xx,yy+bndry_par->dir,zz) = bndry_nv(xx,yy,zz);
	n_solution.ynext(bndry_par->dir)(xx,yy+bndry_par->dir,zz) = bndry_n(xx,yy,zz);
	p_solution.ynext(bndry_par->dir)(xx,yy+bndry_par->dir,zz) = bndry_p(xx,yy,zz);
        nv_solution.ynext(bndry_par->dir)(xx,yy+bndry_par->dir,zz) = bndry_nv(xx,yy,zz);


      }
    }







    
    
    Field3D v = div_all(nv_solution,n);

    // Calculate sound speed
    Field3D cs = sqrt(mul_all(gamma,div_all(p,n)));

    Field3D n_v = mul_all(n,v);
    Field3D p_v = mul_all(p,v);
    Field3D nv_v = mul_all(nv_solution,v);
    
    ddt(n) = 0.0;
    ddt(p) = 0.0;
    ddt(nv) = 0.0;

    ddt(n) = -Div_par(n_v);

    // Pressure equation
    ddt(p) = -Div_par(p_v) - (gamma - 1.0) * p * Div_par(v);

    // Momentum equation
    //ddt(nv) = -Div_parP_n(n,v,cs) - Grad_par(p) ;
    //ddt(nv) = -Div_par(nv_v) - Grad_par(p);
    if(dissipation){
      ddt(nv) += Div_par_K_Grad_par(Diss,nv);
      //ddt(p) += Div_par_K_Grad_par(Diss,p);
      //ddt(n) += Div_par_K_Grad_par(Diss,n);
    }
    return 0;
  }
};

BOUTMAIN(fluid); // Create a main() function


