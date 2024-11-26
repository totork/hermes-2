



class Loki;

#ifndef __Loki_H__
#define __Loki_H__



#include <bout/physicsmodel.hxx>
#include <field_factory.hxx>
#include <bout/derivs.hxx>
#include "parallel_boundary_region.hxx"
#include "boundary_region.hxx"
#include "../../div_ops.hxx"
#include <algorithm> // For std::max

#include <bout/constants.hxx>
#include <bout/mask.hxx> 
#include <initializer_list>
#include <bout/fv_ops.hxx>


namespace FCI {
class dagp_fv;
};



class Loki : public PhysicsModel {
public:
  virtual ~Loki() {}
protected:
  int init(bool restarting);
  int rhs(BoutReal t);
  
private:
  // The variables that should be evolved                                                                                                                                                                  
  Field3D Ne, NVi, Pe, Pi, VePsi, Vort;
  FieldGroup EvolvingVars;

  // Solution Variables
  Field3D Ne_solution, Ne_source, Ne_bndry;
  // Switched for what fields to evolve                                                                                                                                                                    
  bool evolve_Ne, evolve_NVi, evolve_Pe, evolve_Pi, evolve_VePsi, evolve_Vort;

  //Switches for all terms in the Ne equation                                                                                                                                                              

  bool Ne_ExB, Ne_diamagnetic, Ne_vpar, Ne_gradpar ,Ne_collisional ,Ne_diffusion_perp, Ne_diffusion_par , Ne_sources, Ne_D2DX2,Ne_D2DZ2;

  //Additional (important) variables                                                                                                                                                                       

  Field3D Te,Ti,Ve,Vi,phi,Apar;

  // Diffusion variables                                                                                                                                                                                   

  Field3D D_perp,D_par;

  // Support variables
  bool upwind;
  bool diffusion_perp_FV;
  Field3D xl,yl,zl;
  Field3D g_22;
  Field3D RR,ZZ,theta,rho;
  Field3D x_val;
  //////////////////////////////////////////////////////////////////
  
  //                         New operators

  std::unique_ptr<FCI::dagp_fv> _FCIDiv_a_Grad_perp;
  Field3D FCIDiv_a_Grad_perp(const Field3D &a, const Field3D &f);

  
  
};


const BoutReal e0  = 8.854e-12;      // Permittivity of free space
const BoutReal mu0 = 4.e-7*PI;       // Permeability of free space
const BoutReal qe  = 1.602e-19;      // Electron charge
const BoutReal Me  = 9.109e-31;      // Electron mass
const BoutReal Mp  = 1.67262158e-27; // Proton mass











#endif // __Loki_H__
