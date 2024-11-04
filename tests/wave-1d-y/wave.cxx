//#include <bout/derivs.hxx>

#include <bout/physicsmodel.hxx>
#include <field_factory.hxx>
#include "parallel_boundary_region.hxx"
#include "boundary_region.hxx"



class Wave1D : public PhysicsModel {
private:
  Field3D f, g; // Evolving variables
  Field3D f_solution, g_solution, f_source, g_source;
  Field3D this_y;
  Field3D a;
  Field3D bndry_f,bndry_g;
protected:
  int init(bool restarting) override {
    auto& opt = Options::root();
    auto& optf = opt["f"];
    auto& optg = opt["g"];
    f_solution = optf["solution"].withDefault(Field3D{0.0});
    g_solution = optg["solution"].withDefault(Field3D{0.0});
    f_source = optf["source"].withDefault(Field3D{0.0});
    g_source = optg["source"].withDefault(Field3D{0.0});
    this_y = opt["this_y"].withDefault(Field3D{0.0});
    SAVE_ONCE(this_y);
    SAVE_REPEAT(f_solution, g_solution, f_source, g_source,a);
    // Tell BOUT++ to solve f and g
    solver->add(f, "f");
    solver->add(g, "g");
    a = 0.0;
    bndry_f = 0.0;
    bndry_g = 0.0;
    SAVE_REPEAT(bndry_f,bndry_g);
    return 0;
  }

  int rhs(BoutReal time) override {
    f_solution = this_y - sin(time)*cos(0.5*this_y) + cos(this_y);
    g_solution = this_y*this_y + sin(this_y) + cos(time)*cos(0.1*this_y*this_y);
    
    f_source = 0.2*this_y*sin(0.1*this_y*this_y)*cos(time) - 2*this_y - cos(time)*cos(0.5*this_y) - cos(this_y);
    g_source = -0.5*sin(time)*sin(0.5*this_y) - sin(time)*cos(0.1*this_y*this_y) + sin(this_y) - 1.0;
    
    f.applyBoundary();
    g.applyBoundary();
    
    mesh->communicate(f, g);
    for (const auto &bndry_par :
           mesh->getBoundariesPar()) {
      for (bndry_par->first(); !bndry_par->isDone(); bndry_par->next()) {
	int xx = bndry_par->ind().x();
        int yy = bndry_par->ind().y();
        int zz = bndry_par->ind().z();
	BoutReal f_parvalue = 0.0;
	BoutReal g_parvalue = 0.0;
	if (bndry_par->dir > 0.0){
	  //Parallel boundary in the forward parallel direction
	  //Use cells to interpolate value
	  bndry_f(xx,yy,zz) = (f_solution(xx,yy+1,zz)+f_solution(xx,yy,zz))/2.0;
	  bndry_g(xx,yy,zz) = (g_solution(xx,yy+1,zz)+g_solution(xx,yy,zz))/2.0;
	} else {
	  bndry_f(xx,yy,zz) = (f_solution(xx,yy-1,zz)+f_solution(xx,yy,zz))/2.0;
	  bndry_g(xx,yy,zz) = (g_solution(xx,yy-1,zz)+g_solution(xx,yy,zz))/2.0;
	}
	
	f.ynext(bndry_par->dir)(xx,yy+bndry_par->dir,zz) = bndry_f(xx,yy,zz);
	g.ynext(bndry_par->dir)(xx,yy+bndry_par->dir,zz) = bndry_g(xx,yy,zz);

      }
    }
    //Set the parallel boundary conditions

    

    //Time evolution
    
    ddt(f) = Div_par(g)  ;
    ddt(g) = Div_par(f)  ;
    
    return 0;
  }
};

BOUTMAIN(Wave1D); // Create a main() function
