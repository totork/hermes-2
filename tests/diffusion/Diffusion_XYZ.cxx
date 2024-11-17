//#include <bout/derivs.hxx>

#include <bout/physicsmodel.hxx>
#include <field_factory.hxx>
#include <bout/derivs.hxx>
#include "parallel_boundary_region.hxx"
#include "boundary_region.hxx"
#include "../../div_ops.hxx"


class Diffusion_XYZ : public PhysicsModel {
private:
  Field3D N,N_solution;
  Field3D Dx,Dy,Dz;
  Field3D xl, yl, zl;
  Field3D bndry_N;
  Field3D N_yup,N_ydown;
  Field3D g_22s;
  bool is_mms;
  bool new_operator;
protected:
  int init(bool restarting) override {
    auto& opt = Options::root();
    Dx = opt["Dx"].withDefault(Field3D{0.0});
    Dy = opt["Dy"].withDefault(Field3D{0.0});
    Dz = opt["Dz"].withDefault(Field3D{0.0});

    Dy.applyBoundary("neumann_o2");
    mesh->communicate(Dy);
    Dy.applyParallelBoundary("parallel_neumann_o2");
    
    new_operator = opt["new_operator"].withDefault<bool>(false);
    
    xl = opt["N"]["xl"].withDefault(Field3D{0.0});
    yl = opt["N"]["yl"].withDefault(Field3D{0.0});
    zl = opt["N"]["zl"].withDefault(Field3D{0.0});

    is_mms = opt["solver"]["mms"].withDefault<bool>(false);

    auto coord = N.getCoordinates();
    g_22s = coord->g_22;
    N_solution = 0.0;
    bndry_N = 0.0;
    N_yup = 0.0;
    N_ydown = 0.0;
    SAVE_REPEAT(N_solution,bndry_N,N_yup,N_ydown);
    SAVE_ONCE(Dx,Dy,Dz,xl,yl,zl);
    SOLVE_FOR(N);
    return 0;
  }

  int rhs(BoutReal t) override {

    mesh->communicate(N);

    
    // Calculate the solution of N

    //N_solution = 0.9*yl + 0.2 * sin(5.0 * yl*yl - 2.0*zl) + sin(7*xl + 1.234)*cos(xl)*cos(10.0*zl) + 0.9;

    N_solution = 0.9 + 0.9 * yl + 0.2 * cos(10.0 * t)*sin(5.0 * yl * yl);
    
    //N_solution = 0.9 + 0.9 * yl + cos(xl) + 0.2*cos(10.0 * t) * sin(5.0 * yl*yl - 2.0*zl);
    // N_solution = 0.9 + 0.9 * yl + 0.2*sin(5.0*yl*yl);
    // Apply parallel boundary conditions by hand
    if (is_mms){
      for (const auto &bndry_par :
	     mesh->getBoundariesPar()) {
	for (const auto &pnt : *bndry_par) {
          int xx = pnt.ind().x();
          int yy = pnt.ind().y();
	  int zz = pnt.ind().z();
	  BoutReal N_parvalue = 0.0;
	  if (bndry_par->dir > 0.0){
	    //Parallel boundary in the forward parallel direction
	    //Use cells to interpolate value
	    //bndry_N(xx,yy,zz) = (N_solution(xx,yy+1,zz)+N_solution(xx,yy,zz))/2.0;
	    bndry_N(xx,yy,zz) = (N_solution(xx,yy+1,zz));
	  } else {
	    //bndry_N(xx,yy,zz) = (N_solution(xx,yy-1,zz)+N_solution(xx,yy,zz))/2.0;
	    bndry_N(xx,yy,zz) = (N_solution(xx,yy-1,zz));
	  }
	  
	  N.ynext(bndry_par->dir)(xx,yy+bndry_par->dir,zz) = bndry_N(xx,yy,zz);
	}
      }
    } else {
      // N.applyParallelBoundary("parallel_neumann_o2");
    }

    
    /*
    N_yup = N.yup();
    N_ydown = N.ydown();
    */

    for (const auto& ind : N.getRegion("RGN_NOBNDRY")) {
      N_yup[ind] = N.yup()[ind.yp()];
      N_ydown[ind] = N.ydown()[ind.ym()];
    }

    
    ddt(N) = 0.0;
    if (!new_operator){
      ddt(N) += Div_par_K_Grad_par(Dy,N);
    } else {
      
      BOUT_FOR(i, N.getRegion("RGN_ALL")) {
	ddt(N)[i] = Dy[i]*(N.yup()[i.yp()] - N[i]) / sqrt(g_22s[i]);
      }
      
      //ddt(N) = Div_par(N);
      
    }
    return 0;
  }
};

BOUTMAIN(Diffusion_XYZ); // Create a main() function
