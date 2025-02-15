/*!
 *
 */

#include "neutral-model.hxx"

#include "diffusion2d.hxx"
#include "full-velocity.hxx"
#include "mixed.hxx"
#include "none.hxx"
#include "recycling.hxx"

using bout::globals::mesh;

NeutralModel *NeutralModel::create(Solver *solver, Mesh *mesh,
                                   Options &options) {
  // Decide which neutral model to use
  std::string type  = options["type"].withDefault<std::string>("none");
  options["bulk"].setConditionallyUsed();

  if (type == "none") {
    // Neutral model which does nothing
    return NULL; // new NeutralNone(solver, mesh, options);
  } else if (type == "diffusion2d") {
    // Diffusion in X-Z only
    return new Diffusion2D(solver, mesh, options);
  } else if (type == "recycling") {
    // Recycling at target, assumes exponential neutral profile
    // return new NeutralRecycling(solver, mesh, options);
    throw BoutException("not implemented");
  } else if (type == "fullvelocity") {
    // 3D Navier-Stokes
    throw BoutException("not implemented");
    // return new FullVelocity(solver, mesh, options);
  } else if (type == "mixed") {
    // Diffusive in X-Z, fluid in Y. Similar to UEDGE
    throw BoutException("not implemented");
    // return new NeutralMixed(solver, mesh, options);
  }
  throw BoutException("Unrecognised neutral model '%s'", type.c_str());
}

/*!
 * Atomic processes
 *
 * This code integrates atomic cross-sections over each grid cell
 *
 * NOTE: Currently this only integrates in Y, but should integrate in 3D
 */
void fci_neutral_rates(
    const Field3D &Ne, const Field3D &Te, const Field3D &Ti,
    const Field3D &Vi, // Plasma quantities
    const Field3D &Nn, const Field3D &Tn, const Field3D &Vnpar, // Neutral gas
    Field3D &S, Field3D &F, Field3D &Qi, Field3D &R, // Transfer rates
    Field3D &Riz, Field3D &Rrc, Field3D &Rcx,
    BoutReal NormT, BoutReal NormN, BoutReal NormB, BoutReal NormL, BoutReal NormF,
    bool ionizationloss , Field3D& Dnn) {      // Rates


  UpdatedRadiatedPower hydrogen;
  // Allocate output fields
  S = 0.0;
  F = 0.0;
  Qi = 0.0;
  R = 0.0;

  Riz = 0.0;
  Rrc = 0.0;
  Rcx = 0.0;

  Coordinates *coord = mesh->getCoordinates();

  for (auto &ind : Ne.getRegion("RGN_NOBNDRY")){
  
        // Integrate rates over each cell in Y
        // NOTE: This should integrate over (x,y,z)

    const auto iyp = ind.yp();
    const auto iym = ind.ym();
    
        // Calculate cell centre (C), left (L) and right (R) values
    BoutReal Te_C = Te[ind],
      Te_L = 0.5 * (Te.ydown()[iym] + Te[ind]),
      Te_R = 0.5 * (Te[ind] + Te.yup()[iyp]);

    BoutReal Ti_C = Ti[ind],
      Ti_L = 0.5 * (Ti.ydown()[iym] + Ti[ind]),
      Ti_R = 0.5 * (Ti[ind] + Ti.yup()[iyp]);

    BoutReal Ne_C = Ne[ind],
      Ne_L = 0.5 * (Ne.ydown()[iym] + Ne[ind]),
      Ne_R = 0.5 * (Ne[ind] + Ne.yup()[iyp]);

    BoutReal Vi_C = Vi[ind],
      Vi_L = 0.5 * (Vi.ydown()[iym] + Vi[ind]),
      Vi_R = 0.5 * (Vi[ind] + Vi.yup()[iyp]);

    BoutReal Tn_C = Tn[ind],
      Tn_L = 0.5 * (Tn.ydown()[iym] + Tn[ind]),
      Tn_R = 0.5 * (Tn[ind] + Tn.yup()[iyp]);

    BoutReal Nn_C = Nn[ind],
      Nn_L = 0.5 * (Nn.ydown()[iym] + Nn[ind]),
      Nn_R = 0.5 * (Nn[ind] + Nn.yup()[iyp]);

    BoutReal Vn_C = Vnpar[ind],
      Vn_L = 0.5 * (Vnpar.ydown()[iym] + Vnpar[ind]),
      Vn_R = 0.5 * (Vnpar[ind] + Vnpar.yup()[iyp]);

    
    if (Ne_C < 0.)
      Ne_C = 0.0;
    if (Ne_L < 0.)
      Ne_L = 0.0;
    if (Ne_R < 0.)
      Ne_R = 0.0;
    if (Nn_C < 0.)
      Nn_C = 0.0;
    if (Nn_L < 0.)
      Nn_L = 0.0;
    if (Nn_R < 0.)
      Nn_R = 0.0;
	
        // Jacobian (Cross-sectional area)

    BoutReal J_C = coord->J[ind],
      J_L = 0.5 * (coord->J.ydown()[iym] + coord->J[ind]),
      J_R = 0.5 * (coord->J[ind] + coord->J.yup()[iyp]);
    

    ///////////////////////////////////////////
    // Charge exchange
    
    BoutReal R_cx_L = Ne_L * Nn_L * hydrogen.chargeExchange(Te_L * NormT) *
      (NormN / NormF);
    BoutReal R_cx_C = Ne_C * Nn_C * hydrogen.chargeExchange(Te_C * NormT) *
      (NormN / NormF);
    BoutReal R_cx_R = Ne_R * Nn_R * hydrogen.chargeExchange(Te_R * NormT) *
      (NormN / NormF);

    // Power transfer from plasma to neutrals
    // Factor of 3/2 to convert temperature to energy

    Qi[ind] = (3. / 2) * (J_L * (Ti_L - Tn_L) * R_cx_L +
 			      4. * J_C * (Ti_C - Tn_C) * R_cx_C +
			      J_R * (Ti_R - Tn_R) * R_cx_R) /
      (6. * J_C);
    
        // Plasma-neutral friction
    F[ind] =
      (J_L * (Vi_L - Vn_L) * R_cx_L + 4. * J_C * (Vi_C - Vn_C) * R_cx_C +
       J_R * (Vi_R - Vn_R) * R_cx_R) /
      (6. * J_C);
    
    // Cell-averaged rate
    Rcx[ind] =
      (J_L * R_cx_L + 4. * J_C * R_cx_C + J_R * R_cx_R) / (6. * J_C);
    
    ///////////////////////////////////////
    // Recombination

    BoutReal R_rc_L = hydrogen.recombination(Ne_L * NormN, Te_L * NormT) *
      SQ(Ne_L) * NormN / NormF;
    BoutReal R_rc_C = hydrogen.recombination(Ne_C * NormN, Te_C * NormT) *
      SQ(Ne_C) * NormN / NormF;
    BoutReal R_rc_R = hydrogen.recombination(Ne_R * NormN, Te_R * NormT) *
      SQ(Ne_R) * NormN / NormF;

    // Radiated power from plasma
    // Factor of 1.09 so that recombination becomes an energy source at
    // 5.25eV
    R[ind] = (J_L * (1.09 * Te_L - 13.6 / NormT) * R_rc_L +
		  4. * J_C * (1.09 * Te_C - 13.6 / NormT) * R_rc_C +
		  J_R * (1.09 * Te_R - 13.6 / NormT) * R_rc_R) /
      (6. * J_C);
    
        // Plasma sink / neutral source
    S[ind] =
      (J_L * R_rc_L + 4. * J_C * R_rc_C + J_R * R_rc_R) / (6. * J_C);

        // Transfer of ion momentum to neutrals
    F[ind] += (J_L * Vi_L * R_rc_L + 4. * J_C * Vi_C * R_rc_C +
		   J_R * Vi_R * R_rc_R) /
      (6. * J_C);
    
        // Transfer of ion energy to neutrals
    Qi[ind] += (3. / 2) *
      (J_L * Ti_L * R_rc_L + 4. * J_C * Ti_C * R_rc_C +
       J_R * Ti_R * R_rc_R) /
      (6. * J_C);

        // Cell-averaged rate
    Rrc[ind] =
      (J_L * R_rc_L + 4. * J_C * R_rc_C + J_R * R_rc_R) / (6. * J_C);

        ///////////////////////////////////////
        // Ionisation

    BoutReal R_iz_L =
            Ne_L * Nn_L * hydrogen.ionisation(Te_L * NormT) * NormN / NormF;
    BoutReal R_iz_C =
            Ne_C * Nn_C * hydrogen.ionisation(Te_C * NormT) * NormN / NormF;
    BoutReal R_iz_R =
            Ne_R * Nn_R * hydrogen.ionisation(Te_R * NormT) * NormN / NormF;

        // Neutral sink, plasma source
    S[ind] -=
            (J_L * R_iz_L + 4. * J_C * R_iz_C + J_R * R_iz_R) / (6. * J_C);

        // Transfer of neutral momentum to ions
    F[ind] -= (J_L * Vn_L * R_iz_L + 4. * J_C * Vn_C * R_iz_C +
                       J_R * Vn_R * R_iz_R) /
                      (6. * J_C);

        // Transfer of neutral energy to ions
    Qi[ind] -= (3. / 2) *
                       (J_L * Tn_L * R_iz_L + 4. * J_C * Tn_C * R_iz_C +
                        J_R * Tn_R * R_iz_R) /
                       (6. * J_C);

        // Ionisation and electron excitation energy
    if (ionizationloss){
      R[ind] += (1.0 / NormT) *
	(J_L * R_iz_L + 4. * J_C * R_iz_C + J_R * R_iz_R) /
	(6. * J_C);
    }
        // Cell-averaged rate
    Riz[ind] =
      (J_L * R_iz_L + 4. * J_C * R_iz_C + J_R * R_iz_R) / (6. * J_C);



    BoutReal sigma_cx = Ne[ind] * NormN * hydrogen.chargeExchange(Te[ind]*NormT)/NormF;
	
    BoutReal sigma_iz = Ne[ind] * NormN * Nn[ind] * hydrogen.ionisation(Te[ind]*NormT)/NormF;

    // Neutral diffusion coefficient
    BoutReal vth_n = sqrt(Tn[ind]);
    BoutReal a0 = 3.1415*SQ(5.29e-11);
    BoutReal lambda_nn = 1. / (NormN*Nn[ind]*a0); // meters
    BoutReal Lmax = 0.02;
    if(lambda_nn > Lmax) {
      lambda_nn = Lmax;
    }	
    lambda_nn /= NormL; // Normalised length to Lnorm

    BoutReal sigma_nn = vth_n / lambda_nn;	    
    BoutReal sigma = sigma_cx + sigma_nn + sigma_iz;	
    Dnn[ind] = SQ(vth_n) / sigma;
  }
}
