//#include <bout/derivs.hxx>

#include <bout/physicsmodel.hxx>




class Wave1D : public PhysicsModel {
private:
  Field3D f, g; // Evolving variables
  Field3D f_solution, g_solution, f_source, g_source;

protected:
  int init(bool restarting) override {
    auto& opt = Options::root();
    auto& optf = opt["f"];
    auto& optg = opt["g"];
    f_solution = optf["solution"].withDefault(Field3D{0.0});
    g_solution = optg["solution"].withDefault(Field3D{0.0});
    f_source = optf["source"].withDefault(Field3D{0.0});
    g_source = optg["source"].withDefault(Field3D{0.0});
   
    SAVE_REPEAT(f_solution, g_solution, f_source, g_source);
    // Tell BOUT++ to solve f and g
    solver->add(f, "f");
    solver->add(g, "g");
    
    return 0;
  }

  int rhs(BoutReal UNUSED(t)) override {
    auto& opt = Options::root();
    auto& optf = opt["f"];
    auto& optg = opt["g"];
    f_solution = optf["solution"].withDefault(Field3D{0.0});
    g_solution = optg["solution"].withDefault(Field3D{0.0});
    f_source = optf["source"].withDefault(Field3D{0.0});
    g_source = optg["source"].withDefault(Field3D{0.0});
    
    f.applyBoundary();
    g.applyBoundary();
    mesh->communicate(f, g); // Communicate guard cells

    
    
    //set parallel Boundary conditions
    /*
    for (const auto& ind : f.getRegion("RGN_NOBNDRY")){
      if ((mesh->lastY()) && (ind.y() == mesh->yend)){
	f(ind.x(),ind.y(),ind.z()) = f_solution(ind.x(),ind.y(),ind.z());
	g(ind.x(),ind.y(),ind.z()) = g_solution(ind.x(),ind.y(),ind.z());

	f(ind.x(),ind.y()-1,ind.z()) = f_solution(ind.x(),ind.y()-1,ind.z());
        g(ind.x(),ind.y()-1,ind.z()) = g_solution(ind.x(),ind.y()-1,ind.z());
      }
      if ((mesh->firstY()) && (ind.y() == mesh->ystart)){
	f(ind.x(),ind.y(),ind.z()) = f_solution(ind.x(),ind.y(),ind.z());
        g(ind.x(),ind.y(),ind.z()) = g_solution(ind.x(),ind.y(),ind.z());

	f(ind.x(),ind.y()+1,ind.z()) = f_solution(ind.x(),ind.y()+1,ind.z());
        g(ind.x(),ind.y()+1,ind.z()) = g_solution(ind.x(),ind.y()+1,ind.z());
      }
    }
    */
    g.applyParallelBoundary();
    f.applyParallelBoundary();
    

    //Time evolution
    ddt(f) = Grad_par(g) + f_source;
    ddt(g) = Grad_par(f) + g_source;
    
    return 0;
  }
};

BOUTMAIN(Wave1D); // Create a main() function
