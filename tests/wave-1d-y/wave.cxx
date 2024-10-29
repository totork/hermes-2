#include <bout/derivs.hxx>
#include <bout/field_factory.hxx>
#include <bout/physicsmodel.hxx>
#include <bout/unused.hxx>
#include "div_ops.hxx"
#include <derivs.hxx>
#include <field_factory.hxx>
#include <initialprofiles.hxx>

#include <invert_parderiv.hxx>
#include "parallel_boundary_region.hxx"
#include "boundary_region.hxx"

#include "div_ops.hxx"
#include "loadmetric.hxx"

#include <bout/constants.hxx>
#include <bout/assert.hxx>
#include <bout/fv_ops.hxx>
#include <cmath>


class Wave1D : public PhysicsModel {
private:
  Field3D f, g; // Evolving variables

protected:
  int init(bool UNUSED(restarting)) override {

    // Tell BOUT++ to solve f and g
    solver->add(f, "f");
    solver->add(g, "g");

    return 0;
  }

  int rhs(BoutReal UNUSED(t)) override {
    f.applyBoundary();
    g.applyBoundary();
    mesh->communicate(f, g); // Communicate guard cells
    f.applyParallelBoundary();
    g.applyParallelBoundary();
    // Central differencing
    ddt(f) = Div_par(g);
    ddt(g) = Div_par(f);
    return 0;
  }
};

BOUTMAIN(Wave1D); // Create a main() function
