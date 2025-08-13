/*
    Copyright B.Dudson, J.Leddy, University of York, September 2016
              email: benjamin.dudson@york.ac.uk

    This file is part of Hermes-2 (Hot ion).

    Hermes is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Hermes is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Hermes.  If not, see <http://www.gnu.org/licenses/>.
  
*/

class Hermes;

#ifndef __HERMES_H__
#define __HERMES_H__

#include <bout/physicsmodel.hxx>

#include <bout/invert_laplace.hxx>
#include <bout/invert/laplacexy.hxx>
#include <bout/invert/laplacexz.hxx>
#include <bout/constants.hxx>
#include <bout/mask.hxx>

#include "neutral-model.hxx"

// OpenADAS interface Atomicpp by T.Body
#include "atomicpp/ImpuritySpecies.hxx"

#include "div_ops.hxx"

namespace FCI {
class dagp_fv;
};

class Hermes : public PhysicsModel {
public:
  virtual ~Hermes() {}
protected:
  int init(bool restarting);
  int rhs(BoutReal t);
  
  int precon(BoutReal t, BoutReal gamma, BoutReal delta);
private:
  // Equilibrium current
  Field2D Jpar0;

  BoutReal nesheath_floor; // Density floor used in sheath boundary conditions

  
  // Evolving variables
  Field3D Ne;         // Electron density
  Field3D Pe, Pi;     // Electron and Ion pressures
  Field3D VePsi;      // Combination of Ve and psi
  Field3D Vort;       // Vorticity
  Field3D NVi;        // Parallel momentum

  Field3D Pe_yup, Pe_ydown,kappa_epar_yup,kappa_epar_ydown;
  
  Field3D xl,yl,zl;
  
  FieldGroup EvolvingVars;


  Field3D solution_psi;


  BoutReal lambda_sheath;
  
  // Auxilliary variables
  Field3D Te;         // Electron temperature
  Field3D Ti;         // Ion temperature
  Field3D Ve, Vi, Jpar;  // Electron and ion parallel velocities
  Field3D psi;        // Electromagnetic potential (-A_||)
  Field3D phi;        // Electrostatic potential

  bool use_H3_div_par;
  bool use_conduction_map;
  // DEBUG VARIABLES
  Field3D a;
  Field3D b;
  Field3D d;
  Field3D debug_visheath,debug_VePsisheath,debug_vesheath,debug_sheathexp,debug_tisheath;
  Field3D debug_phisheath;
  Field3D debug_denom,debug_soundspeed;
  Field3D debug_phibndry3d;
  Field3D debug_Pe_conduction_A,debug_Pe_conduction_B;
  Field3D debug_decay_Ne;

  bool inner_VePsi_dirichlet;
  bool inner_NVi_dirichlet;
  bool phi_inneraverage;
  bool phi_dirichlet;
  bool test_profiles;
  
  Field3D Vi_sheath , Ve_sheath , Jpar_sheath , Ne_sheath , Te_sheath , Ti_sheath , Vort_sheath,phi_sheath;
  Field3D Te_ythis,Te_ynext,Te_yprev;
  bool new_sheaths;
  bool sheath_interpolate;
  bool sheath_infsink;
  BoutReal infsink_Te, infsink_amp;
  BoutReal infsink_Ne;
  Field3D debug_sheath_infsink;
  bool damp_core_vorticity, damp_edge_vorticity;
  bool J_equalize;
  bool set_inner_neumann;
  bool check_finite;
  // Limited variables
  Field3D Telim, Tilim;
  Field3D Te32, Ti32;
  bool isMMS;
  bool boundarydecay;
  // Collisional terms
  Field3D nu, kappa_epar, kappa_ipar, Dn,eta_epar;
  BoutReal tau_e0, tau_i0;
  Field3D tau_e, tau_i;          // Collision times for electrons and ions
  Field3D Wi;                    // Energy transfer from electrons to ions
  Field3D Pi_ciperp, Pi_cipar, Pi_ci;   // Ion collisional stress tensor
  BoutReal resistivity_multiply; ///< Factor in front of nu

  Field3D classical_mu_vort;
  
  BoutReal flux_limit_alpha;  // Flux limiter. < 0 disables
  BoutReal kappa_limit_alpha; // Heat flux limiter from SOLPS
  BoutReal eta_limit_alpha;   // Momentum flux limiter from SOLPS
  BoutReal floor_eta_epar;
  BoutReal scale_ExB;
  Field3D eta_limit_denom;
  BoutReal floor_kappa_epar,floor_kappa_ipar;
  Field3D boundary_direction;
  BoutReal floor_Ne,floor_Te,floor_Ti;
  Field3D Te_yup,Te_ydown;
  Field3D Ne_ym2,Ne_ym1,Ne_yp1,Ne_yp2;
  Field3D NVi_ym2, NVi_ym1, NVi_yp1, NVi_yp2;
  Field3D Te_ym2, Te_ym1, Te_yp1, Te_yp2;
  BoutReal floor_vel;
  
  // Switches for evolving variables
  bool evolve_plasma;   // Should plasma be evolved?
  bool show_timesteps;  // Show intermediate timesteps?
  bool evolve_te;       // Evolve electron temperature?
  bool evolve_ti;       // Evolve ion temperature?
  bool evolve_vort;     // Evolve vorticity?
  bool evolve_ni;       // Evolve ion density instead?
  bool evolve_nvi;
  bool evolve_vepsi;
  bool evolve_ne;
  bool electromagnetic; // Include magnetic potential psi
  bool FiniteElMass;    // Finite Electron Mass

  Field3D oness,zeroes;


  bool radial_buffers;
  int radial_inner_width;
  int radial_outer_width;
  BoutReal radial_buffer_D;

  BoutReal low_diffuse_value;
  // All variables for the rhie chow velocity correction
  
  bool use_rhie_interpolation;
  Field3D rhie_cor_up,rhie_cor_down;
  /////////////////////////////////////////////////////
  // Switches for all the terms in the equations

  // Density equation
  bool Ne_ExB, Ne_mag, Ne_parflow, Ne_collision, Ne_anomalous, Ne_sources,Ne_hyper, Ne_numdiff, Ne_lowdiffuse;

  // Ion momentum
  bool NVi_ExB, NVi_mag, NVi_parflow, NVi_parpressure, NVi_parviscos, NVi_collision, NVi_anomalous,NVi_hyper,NVi_numdiff; 
  bool NVi_supsonicdampening;
  
  // Electron pressure
  bool Pe_ExB, Pe_mag, Pe_parflow, Pe_conduction, Pe_ohmic, Pe_thermalforce, Pe_thermalcurrent;
  bool Pe_collision, Pe_anomalous, Pe_sources, Pe_energyexchange,Pe_hyper,Pe_numdiff;
  bool Pe_dampening, Pe_lowdiffuse;
  // Ion Pressure
  bool Pi_ExB, Pi_mag, Pi_parflow, Pi_conduction, Pi_diamagenergyexchange, Pi_parviscousheat;
  bool Pi_resistivedrift, Pi_perpviscous, Pi_sources,Pi_hyper,Pi_numdiff,Pi_anomalous, Pi_energyexchange, Pi_lowdiffuse;

  // Vorticity
  bool Vort_mag, Vort_parcurrent, Vort_polarcurrent, Vort_collision, Vort_parviscous;
  bool Vort_anomalous,Vort_hyper,Vort_numdiff, Vort_parflow, Vort_dissipation, Vort_sheathdissipation, Vort_dissipation_par;
  bool Vort_phidissipation, Vort_anomalous_par;
  bool sheathdissipation_espeed;
  bool Vort_dissipation_espeed;
  bool poloidal_flows;
  bool Vort_dirichlet;
  // Electron velocity
  bool VePsi_parefield, VePsi_parpressure, VePsi_partemp, VePsi_parcurrent, VePsi_ExB, VePsi_parflow,VePsi_hyper,VePsi_numdiff;
  bool VePsi_parallelvisc,VePsi_supsonicdampening, VePsi_anomalous, VePsi_dissipation, VePsi_sheathdissipation;

  Field3D Vort_diss;
  
  
  // Field for the terms
  bool TE_Ne,TE_NVi,TE_Pe,TE_Pi,TE_Vort,TE_VePsi;

  Field3D TE_Ne_ExB, TE_Ne_mag, TE_Ne_parflow, TE_Ne_collision, TE_Ne_anomalous, TE_Ne_sources, TE_Ne_hyper, TE_Ne_numdiff, TE_Ne_lowdiffuse;
  
  // Fields for ion momentum terms
  Field3D TE_NVi_ExB, TE_NVi_mag, TE_NVi_parflow, TE_NVi_parpressure, TE_NVi_parviscos, TE_NVi_collision, TE_NVi_anomalous, 
    TE_NVi_hyper, TE_NVi_numdiff;
  Field3D TE_NVi_supsonicdampening;
  // Fields for electron pressure terms
  Field3D TE_Pe_ExB, TE_Pe_mag, TE_Pe_parflow, TE_Pe_conduction, TE_Pe_ohmic, TE_Pe_thermalforce, TE_Pe_thermalcurrent;
  Field3D TE_Pe_collision, TE_Pe_anomalous, TE_Pe_sources, TE_Pe_energyexchange, TE_Pe_hyper, TE_Pe_numdiff, TE_Pe_lowdiffuse;
  Field3D TE_Pe_dampening,TE_Pe_sheath;
  // Fields for ion pressure terms
  Field3D TE_Pi_ExB, TE_Pi_mag, TE_Pi_parflow, TE_Pi_conduction, TE_Pi_diamagenergyexchange, TE_Pi_parviscousheat;
  Field3D TE_Pi_resistivedrift, TE_Pi_perpviscous, TE_Pi_sources, TE_Pi_hyper, TE_Pi_numdiff,TE_Pi_anomalous, TE_Pi_energyexchange, TE_Pi_lowdiffuse;

  // Fields for vorticity terms
  Field3D TE_Vort_mag, TE_Vort_parcurrent, TE_Vort_polarcurrent, TE_Vort_collision, TE_Vort_parviscous;
  Field3D TE_Vort_anomalous, TE_Vort_anomalous_par, TE_Vort_hyper, TE_Vort_numdiff, TE_Vort_parflow, TE_Vort_dissipation, TE_Vort_sheathdissipation;
  Field3D TE_Vort_dissipation_par, TE_Vort_phidissipation;
  // Fields for electron velocity terms
  Field3D TE_VePsi_parefield, TE_VePsi_parpressure, TE_VePsi_partemp, TE_VePsi_parcurrent, TE_VePsi_ExB, TE_VePsi_parflow;
  Field3D TE_VePsi_hyper, TE_VePsi_numdiff , TE_VePsi_parallelvisc, TE_VePsi_supsonicdampening, TE_VePsi_anomalous, TE_VePsi_sheathdissipation;
  Field3D TE_VePsi_dissipation;
  
  //////////////////////////////////////////////////////
  
  bool j_pol_pi;       // Polarisation current with explicit Pi dependence
  bool j_pol_simplified;       // Polarisation current with explicit Pi dependence
  bool calc_potential;
  bool resistivity; // Resistivity: Psi -> Pe
  bool use_Div_n_bxGrad_f_B_XPPM; //Use stencil operator for ExB
  bool use_bracket;                 //Use the bracket for the curvature drifts
  bool norm_dxdydz;
  bool use_Div_parP_n;
  bool use_new_conduction;
  bool use_new_div_par;
  bool use_new_grad_par;
  bool use_new_divagradperp;
  bool use_new_viscosity;
  bool use_slope_limiter;
  bool use_Delp2;
  bool use_Te_limiter;
  bool use_Ti_limiter;
  bool use_Ve_limiter;
  bool use_viscosity_limiter;
  bool use_conduction_limiter;
  BoutReal Te_limiter_value;
  BoutReal Ti_limiter_value;
  BoutReal Ve_limiter_value;
  BoutReal viscosity_limiter_value;
  BoutReal conduction_limiter_value;
  bool use_Vi;
  
  // Anomalous perpendicular diffusion coefficients
  BoutReal anomalous_D;    // Density diffusion
  BoutReal anomalous_chi;  // Electron thermal diffusion
  BoutReal anomalous_nu;   // Momentum diffusion (kinematic viscosity)
  BoutReal anomalous_nu_par;
  Field3D hyper_D, hyper_chi, hyper_nu;
  Field3D num_D, num_nu, num_chi;
  Field3D num_Vort, num_VePsi;
  
  Field3D a_d3d, a_chi3d, a_nu3d, a_nu3d_par; // 3D coef
  bool anomalous_D_nvi; // Include terms in momentum equation
  bool anomalous_D_pepi; // Include terms in Pe, Pi equations
  

  bool phi3d;         // Use a 3D solver for phi
  

  bool boussinesq;     // Use a fixed density (Nnorm) in the vorticity equation

  bool Div_parP_n_sheath_extra{true}; // Use special handling for the sheath

  // Sheath heat transmission factor
  int sheath_model;     // Sets boundary condition model
  BoutReal sheath_gamma_e, sheath_gamma_i;  // Heat transmission
  BoutReal neutral_vwall; // Scale velocity at the wall
  bool sheath_yup, sheath_ydown; 
  bool test_boundaries;
  bool sheath_allow_supersonic; // If plasma is faster than sound speed, go to plasma velocity
  bool parallel_sheaths;  
  int par_sheath_model;  // Sets parallel boundary condition model
  BoutReal electron_weight;  // electron heaviness in units of m_e (for slower boundaries)
  bool par_sheath_ve;
  Field3D sheath_dpe, sheath_dpi; 

  Field3D fastest_ispeed,fastest_espeed;
  
  BoundaryRegionPar* bndry_par;

  Field2D wall_flux; // Particle flux to wall (diagnostic)
  Field2D wall_power; // Power flux to wall (diagnostic)
  
  // Fix density in SOL
  bool sol_fix_profiles;
  std::shared_ptr<FieldGenerator> sol_ne, sol_te; // Generating functions
  
  // Output switches for additional information
  bool verbose;    // Outputs additional fields, mainly for debugging
  bool output_ddt; // Output time derivatives


  // neutral variables
  bool evolve_neutrals, evolve_pn;
  Field3D Nn;
  Field3D NnVn;
  Field3D Pn;
  Field3D Tn;
  Field3D Vn;

  bool neutralplasmainteraction;
  bool simplified_diffusion;
  
  BoutReal floor_Nn,floor_Tn;
  BoutReal anomalous_Dn;
  BoutReal Recycling_coef;
  Field3D Recycling_flux;
  BoutReal neutrals_lmax;
  Field3D Sneutral, Fn, Qin, Rn, Riz, Rrc, Rcx, Dnn;
  
  bool TE_Nn;
  bool Nn_parflow, Nn_perpflow, Nn_sources, Nn_hyper, Nn_numdiff;
  Field3D TE_Nn_parflow, TE_Nn_perpflow, TE_Nn_sources, TE_Nn_hyper, TE_Nn_numdiff;


  bool TE_NnVn;
  bool NnVn_parflow, NnVn_perpflow, NnVn_pargradient, NnVn_pardiffusion, NnVn_friction, NnVn_hyper, NnVn_numdiff;
  Field3D TE_NnVn_parflow, TE_NnVn_perpflow, TE_NnVn_pargradient, TE_NnVn_pardiffusion, TE_NnVn_friction, TE_NnVn_hyper, TE_NnVn_numdiff;


  bool TE_Pn;
  bool Pn_parflow, Pn_perpflow, Pn_parcompression, Pn_perpdiffusion, Pn_sources, Pn_hyper, Pn_numdiff;
  Field3D TE_Pn_parflow, TE_Pn_perpflow, TE_Pn_parcompression, TE_Pn_perpdiffusion, TE_Pn_sources, TE_Pn_hyper, TE_Pn_numdiff;
  
  // Numerical dissipation


  BoutReal Pe_dampening_Te;
  BoutReal Pe_dampening_factor;

  bool NVi_supsonic_dissipation;
  BoutReal NVi_supsonic_factor;
  Field3D NVi_dampening;
  BoutReal NVi_supsonic_cut;

  bool Ve_supsonic_dissipation;
  BoutReal Ve_supsonic_factor;
  BoutReal Ve_supsonic_cut;
  Field3D Ve_dampening;
  
  Field3D qi;
  // Sources and profiles
  
  bool ramp_mesh;   // Use Ne,Pe in the grid file for starting ramp target
  BoutReal ramp_timescale; // Length of time for the initial ramp

  bool sheath_ramp;
  BoutReal sheath_ramp_time,sheath_ramp_factor;

  BoutReal Ne_target, Te_target, Ti_target,adaptive_overshoot;
  
  bool adapt_source; // Use a PI controller to feedback profiles
  bool core_sources; // Sources only in the core
  bool energy_source; // Add the same amount of energy to each particle
  BoutReal source_p, source_i;  // Proportional-Integral controller
  Coordinates::FieldMetric Sn, Spe, Spi; // Sources in density, Pe and Pi
  Field3D NeSource, PeSource, PiSource, VortSource; // Actual sources added
  bool density_inflow;  // Does incoming density have momentum?
  
  bool source_vary_g11; // Multiply source by g11
  Coordinates::FieldMetric g11norm;

  Field3D SQSQ_g_11,SQSQ_g_33;
  // Boundary fluxes

  bool pe_bndry_flux;   // Allow flux of pe through radial boundaries
  bool ne_bndry_flux;   // Allow flux of ne through radial boundaries
  bool vort_bndry_flux; // Allow flux of vorticity through radial boundaries
  
  // Normalisation parameters
  BoutReal Tnorm, Nnorm, Bnorm;
  BoutReal AA, Cs0, rho_s0, Omega_ci;
  BoutReal mi_me, me_mi, beta_e;
  
  // Curvature, Grad-B drift
  Vector3D Curlb_B; // Curl(b/B)

  Vector3D bxcv;
  Field3D bxcvx,bxcvy,bxcvz;
  
  // Perturbed parallel gradient operators
  Field3D Grad_parP(const Field3D &f);
  Field3D Div_parP(const Field3D &f, const bool newop);
  Field3D Div_parP_f(const Field3D &f, const Field3D &v, Field3D &cs);
  Field3D Div_parP_n(const Field3D &f, const Field3D &v, Field3D &cs,
                     const BoutMask &fwd, const BoutMask &bwd);

  Field3D hyperdissipation(const Field3D &a, const Field3D &b);
  Field3D numericaldissipation(const Field3D &a, const Field3D &b);
  Field3D term_limiter(const Field3D &a, const BoutReal &val);
  // Electromagnetic solver for finite electron mass case
  bool split_n0_psi;   // Split the n=0 component of Apar (psi)?
  //Laplacian *aparSolver;
  // LaplaceXZ *aparSolver;
  std::unique_ptr<LaplaceXZ> aparSolver{nullptr};
  //  std::unique_ptr<LaplaceXZ> aparSolver{nullptr};
  // std::unique_ptr<LaplaceXY> aparXY{nullptr};
  LaplaceXY *aparXY;    // Solves n=0 component
  Field2D psi2D;        // Axisymmetric Psi
  
  // Solvers for the electrostatic potential
  bool use_old_aparSolver;
  bool split_n0;        // Split solve into n=0 and n~=0?
  // std::unique_ptr<LaplaceXY> laplacexy{nullptr};
  LaplaceXY *laplacexy; // Laplacian solver in X-Y (n=0)
  Field2D phi2D;        // Axisymmetric phi

  bool phi_boundary_relax; ///< Relax the boundary towards Neumann?
  bool phi_sheath_relax;
  bool phi_relaxation;
  BoutReal phi_relaxation_time;
  BoutReal phi_boundary_timescale; ///< Relaxation timescale
  BoutReal phi_boundary_last_update; ///< The last time the boundary was updated
  BoutReal phi_sheath_timescale;
  BoutReal phi_sheath_last_update;
  bool anomalous_precon;
  bool newXZsolver; 
  std::unique_ptr<Laplacian> phiSolver{nullptr}; // Old Laplacian in X-Z
  std::unique_ptr<LaplaceXZ> newSolver{nullptr};
  //std::unique_ptr<LaplaceXZ> newaparSolver{nullptr};
  std::unique_ptr<Laplacian> neutralSolver{nullptr};
  std::unique_ptr<Laplacian> preconSolver{nullptr};

  std::unique_ptr<Laplacian> oldaparSolver{nullptr};
  bool relaxation;
  Field3D phi_1;
  BoutReal lambda_0,lambda_2;
  
  // Mesh quantities
  Coordinates::FieldMetric B12, B32, B42;

  bool fci_transform;
  Field3D Bxyz, logB, Bxy;
  Field3D Bxz, logBxz;
  Field3D bracket_factor;
  Field3D fci_curvature(const Field3D &f, const bool &bool_bracket);

  
  BoutMask fwd_bndry_mask, bwd_bndry_mask;

  // perp boundary
  BoutReal fall_off_Ne, fall_off_Pe, fall_off_Pi;
  bool fall_off;
  Field3D xdist;

  // new diff methods
  std::unique_ptr<FCI::dagp_fv> _FCIDiv_a_Grad_perp;
  Field3D FCIDiv_a_Grad_perp(const Field3D &a, const Field3D &f);
};

/// Fundamental constants

const BoutReal e0  = 8.854e-12;      // Permittivity of free space
const BoutReal mu0 = 4.e-7*PI;       // Permeability of free space
const BoutReal qe  = 1.602e-19;      // Electron charge
const BoutReal Me  = 9.109e-31;      // Electron mass
const BoutReal Mp  = 1.67262158e-27; // Proton mass

#endif // __HERMES_H__
