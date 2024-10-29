//#include <bout/derivs.hxx>

#include <bout/physicsmodel.hxx>




class Wave1D : public PhysicsModel {
private:
  Field3D f, g,f_solution,g_solution; // Evolving variables

protected:
  int init(bool restarting) override {
    auto& opt = Options::root();
    auto& optf = opt["f"];
    auto& optg = opt["g"];
    f_solution = optf["solution"].withDefault(Field3D{0.0});
    g_solution = optg["solution"].withDefault(Field3D{0.0});
    
    // Tell BOUT++ to solve f and g
    solver->add(f, "f");
    solver->add(g, "g");

    return 0;
  }

  int rhs(BoutReal UNUSED(t)) override {
    f.applyBoundary();
    g.applyBoundary();
    mesh->communicate(f, g); // Communicate guard cells

    //set parallel Boundary conditions

    /*
    for (const auto &bndry_par :
	   mesh->getBoundariesPar(BoundaryParType::all)) {
      for (bndry_par->first(); !bndry_par->isDone(); bndry_par->next()){
	int x = bndry_par->ind().x();
        int y = bndry_par->ind().y();
        int z = bndry_par->ind().z();

	f.ynext(bndry_par->dir)(x,y+bndry_par->dir,z) = f_solution(x, y, z);
	g.ynext(bndry_par->dir)(x,y+bndry_par->dir,z) = g_solution(x, y, z);
      }
    }
    */

    for (const auto& ind : f.getRegion("RGN_NOBNDRY")){
      if ((mesh->lastY()) && (ind.y() == mesh->yend)){
	f(ind.x(),ind.y(),ind.z()) = f_solution(ind.x(),ind.y(),ind.z());
	g(ind.x(),ind.y(),ind.z()) = g_solution(ind.x(),ind.y(),ind.z());
      }
      if ((mesh->firstY()) && (ind.y() == mesh->ystart)){
	f(ind.x(),ind.y(),ind.z()) = f_solution(ind.x(),ind.y(),ind.z());
        g(ind.x(),ind.y(),ind.z()) = g_solution(ind.x(),ind.y(),ind.z());
      }
    }
    

    //Time evolution
    ddt(f) = Div_par(g);
    ddt(g) = Div_par(f);
    //ddt(f) = 0.0;
    //ddt(g) = 0.0;
    return 0;
  }
};

BOUTMAIN(Wave1D); // Create a main() function
