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

#include <invert_laplace.hxx>
#include <bout/invert/laplacexy.hxx>
#include <bout/invert/laplacexz.hxx>
#include <bout/constants.hxx>
#include <mask.hxx>

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

  FieldGroup EvolvingVars;

  // Auxilliary variables
  Field3D Te;         // Electron temperature
  Field3D Ti;         // Ion temperature
  Field3D Ve, Vi, Jpar;  // Electron and ion parallel velocities
  Field3D psi;        // Electromagnetic potential (-A_||)
  Field3D phi;        // Electrostatic potential

  
  // DEBUG VARIABLES
  Field3D a;
  Field3D b;
  Field3D d;
  Field3D debug_visheath,debug_VePsisheath,debug_vesheath,debug_sheathexp;
  Field3D debug_phisheath;
  Field3D TE_VePsi_pe_par,TE_VePsi_resistivity,TE_VePsi_anom,TE_VePsi_j_par,TE_VePsi_thermal_force,TE_VePsi_par_adv,TE_VePsi_hyper,TE_VePsi_perp,TE_VePsi_numdiff;
  Field3D TE_Ne_ExB, TE_Ne_parflow, TE_Ne_anom, TE_Ne_dia, TE_Ne_hyper;
  Field3D TE_Pe_ExB, TE_Pe_parflow, TE_Pe_anom, TE_Pe_dia, TE_Pe_hyper, TE_Pe_energ_balance, TE_Pe_cond, TE_Pe_thermal_flux, TE_Pe_ohmic, TE_Pe_thermal_force, TE_Pe_par_p_term, TE_Pe_numdiff;
  Field3D TE_NVi_ExB, TE_NVi_dia, TE_NVi_parflow,TE_NVi_pe_par,TE_NVi_viscos,TE_NVi_numdiff,TE_NVi_classical,TE_NVi_hyper,TE_NVi_anom;
  Field3D debug_denom,debug_soundspeed;
  Field3D J_up,J_down,g_11_up,g_11_down,g_22_up,g_22_down,g_33_up,g_33_down,g_13_up,g_13_down,g_12_down,g_12_up,g_23_down,g_23_up;
  Field3D vort_dia,vort_ExB,vort_jpar,vort_parflow,vort_anom,vort_hyper,vort_classical,vort_numdiff;
  bool J_equalize;
  // Limited variables
  Field3D Telim, Tilim;

  // Collisional terms
  Field3D nu, kappa_epar, kappa_ipar, Dn;
  BoutReal tau_e0, tau_i0;
  Field3D tau_e, tau_i;          // Collision times for electrons and ions
  Field3D Wi;                    // Energy transfer from electrons to ions
  Field3D Pi_ciperp, Pi_cipar, Pi_ci;   // Ion collisional stress tensor
  BoutReal resistivity_multiply; ///< Factor in front of nu

  BoutReal flux_limit_alpha;  // Flux limiter. < 0 disables
  BoutReal kappa_limit_alpha; // Heat flux limiter from SOLPS
  BoutReal eta_limit_alpha;   // Momentum flux limiter from SOLPS
  BoutReal scale_ExB;
  BoutReal floor_kappa_epar,floor_kappa_ipar;
  // Neutral gas model
  NeutralModel *neutrals; // Handles evolution of neutral gas
  bool neutral_friction;
  BoutReal frecycle;  // Recycling fraction
  BoutReal ion_neutral_rate; // Fixed ion-neutral collision rate
  
  // Impurity radiation
  BoutReal fimp;             // Impurity fraction (of Ne)
  bool impurity_adas;        // True if using ImpuritySpecies, false if using
  ImpuritySpecies *impurity; // Atomicpp impurity
  
  BoutReal carbon_fraction;
  Field3D Rzrad;             // Radiated power
  RadiatedPower *carbon_rad; // Carbon cooling curve
  
  // Switches
  bool evolve_plasma;   // Should plasma be evolved?
  bool show_timesteps;  // Show intermediate timesteps?
  bool evolve_te;       // Evolve electron temperature?
  bool evolve_ti;       // Evolve ion temperature?
  bool evolve_vort;     // Evolve vorticity?
  bool evolve_ni;       // Evolve ion density instead?
  
  bool electromagnetic; // Include magnetic potential psi
  bool FiniteElMass;    // Finite Electron Mass
  
  bool j_diamag;    // Diamagnetic current: Vort <-> Pe
  bool j_par;       // Parallel current:    Vort <-> Psi
  bool j_pol_pi;       // Polarisation current with explicit Pi dependence
  bool j_pol_simplified;       // Polarisation current with explicit Pi dependence
  bool VorticitySource;
  bool phi_bndry_after_solve;
  bool parallel_flow;
  bool parallel_vort_flow;
  bool parallel_flow_p_term; // Vi advection terms in Pe, Pi
  bool pe_par;      // Parallel pressure gradient: Pe <-> Psi
  bool pe_par_p_term; // Includes terms in Pe,Pi equations
  bool resistivity; // Resistivity: Psi -> Pe
  bool thermal_force; // Force due to temperature gradients
  bool electron_viscosity; // Electron parallel viscosity
  bool ion_viscosity;   // Ion viscosity
  bool ion_viscosity_par; // Parallel part of ion viscosity
  bool electron_neutral;   // Include electron-neutral collisions in resistivity
  bool ion_neutral;        // Include ion-neutral collisions in ion collision time
  bool poloidal_flows;  // Include y derivatives in diamagnetic and ExB drifts
  bool thermal_flux;    // Include parallel and perpendicular energy flux from Te gradients
  bool thermal_conduction; // Braginskii electron heat conduction
  bool electron_ion_transfer; // Electron-ion heat transfer
  bool classical_diffusion; // Collisional diffusion, including viscosity
  bool use_Div_n_bxGrad_f_B_XPPM; //Use stencil operator for ExB
  bool use_bracket;                 //Use the bracket for the curvature drifts
  bool norm_dxdydz;
  bool use_Div_parP_n;
  bool conduction_kappagrad;
  bool Ohmslaw_use_ve;
  Field3D NVi_Div_parP_n;

  bool TE_VePsi,TE_Ne,TE_Pe,TE_NVi;
  
  BoutReal MMS_Ne_ParDiff;
  // Anomalous perpendicular diffusion coefficients
  BoutReal anomalous_D;    // Density diffusion
  BoutReal anomalous_chi;  // Electron thermal diffusion
  BoutReal anomalous_nu;   // Momentum diffusion (kinematic viscosity)
  Field3D a_d3d, a_chi3d, a_nu3d; // 3D coef
  Field3D a_MMS3d;
  bool anomalous_D_nvi; // Include terms in momentum equation
  bool anomalous_D_pepi; // Include terms in Pe, Pi equations
  
  bool ion_velocity;  // Include Vi terms

  bool phi3d;         // Use a 3D solver for phi
  
  bool staggered;     // Use staggered differencing along B

  bool boussinesq;     // Use a fixed density (Nnorm) in the vorticity equation

  bool sinks; // Sink terms for running 2D drift-plane simulations
  bool sheath_closure; // Sheath closure sink on vorticity (if sinks = true)
  bool drift_wave;     // Drift-wave closure (if sinks=true)

  bool radial_buffers; // Radial buffer regions
  int radial_inner_width; // Number of points in the inner radial buffer
  int radial_outer_width; // Number of points in the outer radial buffer
  BoutReal radial_buffer_D; // Diffusion in buffer region
  bool radial_inner_averagey; // Average Ne, Pe, Pi fields in Y in inner radial buffer
  bool radial_inner_averagey_vort; // Average vorticity in Y in inner buffer
  bool radial_inner_averagey_nvi; // Average NVi in Y in inner buffer
  bool radial_inner_zero_nvi; // Damp NVi towards zero in inner buffer

  bool Div_parP_n_sheath_extra{true}; // Use special handling for the sheath
  bool VePsi_perp;
  bool phi_smoothing;
  BoutReal phi_sf;
  
  BoutReal resistivity_boundary; // Value of nu in boundary layer
  int resistivity_boundary_width; // Width of radial boundary
  
  Field3D sink_invlpar; // Parallel inverse connection length (1/L_{||}) for
                        // sink terms
  Field2D alpha_dw;

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
  
  BoundaryRegionPar* bndry_par;

  Field2D wall_flux; // Particle flux to wall (diagnostic)
  Field2D wall_power; // Power flux to wall (diagnostic)
  
  // Fix density in SOL
  bool sol_fix_profiles;
  std::shared_ptr<FieldGenerator> sol_ne, sol_te; // Generating functions
  
  // Output switches for additional information
  bool verbose;    // Outputs additional fields, mainly for debugging
  bool output_ddt; // Output time derivatives
  
  // Numerical dissipation

  BoutReal numdiff, hyper, hyperpar; ///< Numerical dissipation
  int low_pass_z; // Fourier filter in Z 
  BoutReal z_hyper_viscos, x_hyper_viscos, y_hyper_viscos; // 4th-order derivatives
  bool low_n_diffuse; // Diffusion in parallel direction at low density
  bool low_n_diffuse_perp; // Diffusion in perpendicular direction at low density
  
  bool bool_ne_hyper, bool_pe_hyper, bool_pi_hyper;
  Field3D ne_hyper, pe_hyper,pi_hyper; // Hyper-diffusion

  bool bool_VePsi_hyper,bool_NVi_hyper;
  Field3D VePsi_hyper, NVi_hyper;

  bool bool_Vort_hyper;
  Field3D Vort_hyper;
  
  BoutReal scale_num_cs; // Scale numerical sound speed
  BoutReal floor_num_cs; // Apply a floor to the numerical sound speed
  bool vepsi_dissipation; // Dissipation term in VePsi equation
  bool vort_dissipation; // Dissipation term in Vorticity equation
  bool phi_dissipation; // Dissipation term in Vorticity equation

  bool NVi_supsonic_dissipation;
  BoutReal NVi_supsonic_factor;
  Field3D NVi_dampening;
  
  BoutReal VePsi_hyperXZ;
  
  BoutReal ne_num_diff;
  BoutReal ne_num_hyper;
  BoutReal vi_num_diff; // Numerical perpendicular diffusion
  BoutReal ve_num_diff; // Numerical perpendicular diffusion
  BoutReal ve_num_hyper; // Numerical hyper-diffusion
  
  // Sources and profiles
  
  bool ramp_mesh;   // Use Ne,Pe in the grid file for starting ramp target
  BoutReal ramp_timescale; // Length of time for the initial ramp
  Field3D NeTarget, PeTarget, PiTarget; // For adaptive sources
  
  bool adapt_source; // Use a PI controller to feedback profiles
  bool core_sources; // Sources only in the core
  bool energy_source; // Add the same amount of energy to each particle
  BoutReal source_p, source_i;  // Proportional-Integral controller
  Coordinates::FieldMetric Sn, Spe, Spi; // Sources in density, Pe and Pi
  Field3D NeSource, PeSource, PiSource, VortSource; // Actual sources added
  bool density_inflow;  // Does incoming density have momentum?
  
  bool source_vary_g11; // Multiply source by g11
  Coordinates::FieldMetric g11norm;
  
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
  Field3D Div_parP(const Field3D &f);
  Field3D Div_parP_f(const Field3D &f, const Field3D &v, Field3D &cs);
  Field3D Div_parP_n(const Field3D &f, const Field3D &v, Field3D &cs,
                     const BoutMask &fwd, const BoutMask &bwd);

  // Electromagnetic solver for finite electron mass case
  bool split_n0_psi;   // Split the n=0 component of Apar (psi)?
  //Laplacian *aparSolver;
  // LaplaceXZ *aparSolver;
  std::unique_ptr<Laplacian> aparSolver{nullptr};

  // std::unique_ptr<LaplaceXY> aparXY{nullptr};
  LaplaceXY *aparXY;    // Solves n=0 component
  Field2D psi2D;        // Axisymmetric Psi
  
  // Solvers for the electrostatic potential

  bool split_n0;        // Split solve into n=0 and n~=0?
  // std::unique_ptr<LaplaceXY> laplacexy{nullptr};
  LaplaceXY *laplacexy; // Laplacian solver in X-Y (n=0)
  Field2D phi2D;        // Axisymmetric phi

  bool phi_boundary_relax; ///< Relax the boundary towards Neumann?
  BoutReal phi_boundary_timescale; ///< Relaxation timescale
  BoutReal phi_boundary_last_update; ///< The last time the boundary was updated
  
  bool newXZsolver; 
  std::unique_ptr<Laplacian> phiSolver{nullptr}; // Old Laplacian in X-Z
  std::unique_ptr<LaplaceXZ> newSolver{nullptr}; // New Laplacian in X-Z


  bool relaxation;
  Field3D phi_1;
  BoutReal lambda_0,lambda_2;
  
  // Mesh quantities
  Coordinates::FieldMetric B12, B32, B42;

  bool fci_transform;
  Field3D Bxyz, logB;
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
