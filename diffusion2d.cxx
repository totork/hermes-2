
#include "diffusion2d.hxx"

#include <bout/constants.hxx>
#include <bout/fv_ops.hxx>
#include "div_ops.hxx"

using bout::globals::mesh;

Diffusion2D::Diffusion2D(Solver *solver, Mesh*, Options &options) : NeutralModel(options) {
  // 2D (X-Z) diffusive model
  // Neutral gas dynamics
  solver->add(Nn, "Nn");
  solver->add(Pn, "Pn");
  
  Dnn = 0.0; // Neutral gas diffusion

  // SAVE_REPEAT(Dnn);

  Lmax = options["Lmax"].doc("Maximum mean free path [m]").withDefault(1.0);
}

void Diffusion2D::update(const Field3D &Ne, const Field3D &Te, const Field3D &UNUSED(Ti), const Field3D &UNUSED(Vi)) {
  
  mesh->communicate(Nn, Pn);

  // Calculate atomic processes
  BOUT_FOR(i, Ne.getRegion("RGN_ALL")) {
    Nn[i] = std::max(Nn[i], 1e-8);
    BoutReal Tn = Pn[i] / Nn[i];
    Tn = std::max(Tn, 0.01 / Tnorm);

    BoutReal Nelim =
        std::max(Ne[i], 1e-19); // Smaller limit for rate coefficients

    // Charge exchange frequency, normalised to ion cyclotron frequency
    BoutReal sigma_cx =
        Nelim * Nnorm * hydrogen.chargeExchange(Te[i] * Tnorm) / Fnorm;

    // Ionisation frequency, normalised to ion cyclotron frequency
    BoutReal sigma_iz =
        Nelim * Nnorm * Nn[i] * hydrogen.ionisation(Te[i] * Tnorm) / Fnorm;

    // Neutral thermal velocity
    BoutReal vth_n = sqrt(Tn); // Normalised to Cs0

    // Neutral-neutral mean free path
    BoutReal a0 = PI * SQ(5.29e-11);
    BoutReal lambda_nn = 1. / (Nnorm * Nn[i] * a0); // meters
    if (lambda_nn > Lmax) {
      // Limit maximum mean free path
      lambda_nn = Lmax;
    }

    lambda_nn /= Lnorm; // Normalised length to Lnorm
    // Neutral-Neutral collision rate, normalised to ion cyclotron frequency
    BoutReal sigma_nn = vth_n / lambda_nn;

    // Total neutral collision frequency, normalised to ion cyclotron frequency
    BoutReal sigma = sigma_cx + sigma_nn + sigma_iz;

    // Neutral gas diffusion
    Dnn[i] = SQ(vth_n) / sigma;

    // Rates
    BoutReal R_rc = SQ(Nelim) *
                    hydrogen.recombination(Nelim * Nnorm, Te[i] * Tnorm) *
                    Nnorm / Fnorm; // Time normalisation
    BoutReal R_iz = Nelim * Nn[i] * hydrogen.ionisation(Te[i] * Tnorm) * Nnorm /
                    Fnorm; // Time normalisation
    BoutReal R_cx = sigma_cx * Nn[i];

    // Plasma sink / neutral source
    S[i] = R_rc - R_iz;

    // Power transfer from plasma to neutrals

    Qi[i] = R_cx * (3. / 2) * (Te[i] - Tn);

    // Power transfer due to ionisation and recombination
    Qi[i] += (3. / 2) * (Te[i] * R_rc - Tn * R_iz);

    // Ion-neutral friction
    Fperp[i] = R_cx    // Charge-Exchange
               + R_rc; // Recombination

    // Radiated power from plasma
    // Factor of 1.09 so that recombination becomes an energy source at 5.25eV
    Rp[i] = (1.09 * Te[i] - 13.6 / Tnorm) * R_rc +
            (Eionize / Tnorm) * R_iz; // Ionisation energy
  }
  mesh->communicate(Dnn, Fperp);
  Dnn.applyParallelBoundary("parallel_neumann_o2");
  Fperp.applyParallelBoundary("parallel_neumann_o2");
  Nn.applyParallelBoundary();
  Pn.applyParallelBoundary();

  // Neutral density
  ddt(Nn) = 
    + S 
    + FCI::Div_a_Grad_perp(Dnn, Nn);
  
  // Neutral pressure
  ddt(Pn) = (2./3)*Qi
    + FCI::Div_a_Grad_perp(Dnn, Pn)
    ;
  
}

void Diffusion2D::precon(BoutReal, BoutReal gamma, BoutReal) {
  // Neutral gas diffusion
  // Solve (1 - gamma*Dnn*Delp2)^{-1} 
  if(!inv) {
    inv = Laplacian::create();
    // Zero value outer boundary
    
    inv->setInnerBoundaryFlags(INVERT_DC_GRAD | INVERT_AC_GRAD);
    
    inv->setCoefA(1.0);
  }
  
  inv->setCoefD(-gamma*Dnn);
  
  ddt(Nn) = inv->solve(ddt(Nn));
  
  ddt(Pn) = inv->solve(ddt(Pn));
}
