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
protected:
  int init(bool restarting) override {
    auto& opt = Options::root();
    Dx = opt["Dx"].withDefault(Field3D{0.0});
    Dy = opt["Dy"].withDefault(Field3D{0.0});
    Dz = opt["Dz"].withDefault(Field3D{0.0});

    xl = opt["N"]["xl"].withDefault(Field3D{0.0});
    yl = opt["N"]["yl"].withDefault(Field3D{0.0});
    zl = opt["N"]["zl"].withDefault(Field3D{0.0});
    
    N_solution = 0.0;
    bndry_N = 0.0;
    SAVE_REPEAT(N_solution,bndry_N);
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
	    bndry_N(xx,yy,zz) = (N_solution(xx,yy+1,zz)+N_solution(xx,yy,zz))/2.0;
	  } else {
	    bndry_N(xx,yy,zz) = (N_solution(xx,yy-1,zz)+N_solution(xx,yy,zz))/2.0;
	  }
	
	  N.ynext(bndry_par->dir)(xx,yy+bndry_par->dir,zz) = bndry_N(xx,yy,zz);
	  
      }
    }
    
    ddt(N) = 0.0;
    ddt(N) += Div_par_K_Grad_par(Dy,N);
    //ddt(N) += Dy * Grad2_par2(N);
    
    //ddt(N) += Dx * D2DX2(N);
    //ddt(N) += Dz * D2DZ2(N);
    return 0;
  }
};

BOUTMAIN(Diffusion_XYZ); // Create a main() function
