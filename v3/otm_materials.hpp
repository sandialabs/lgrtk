#pragma once

#include <cmath>
#include <iostream>

#include <hpc_dimensional.hpp>
#include <hpc_macros.hpp>
#include <hpc_matrix3x3.hpp>
#include <hpc_symmetric3x3.hpp>
#include <j2/hardening.hpp>

namespace lgr {

HPC_ALWAYS_INLINE HPC_DEVICE void neo_Hookean_point(hpc::deformation_gradient<double> const &F, hpc::pressure<double> const K, hpc::pressure<double> const G,
    hpc::stress<double> &sigma, hpc::pressure<double> &Keff, hpc::pressure<double>& Geff, hpc::energy_density<double>& potential)
{
  auto const J = determinant(F);
  auto const Jinv = 1.0 / J;
  auto const Jm13 = 1.0 / std::cbrt(J);
  auto const Jm23 = Jm13 * Jm13;
  auto const Jm53 = (Jm23 * Jm23) * Jm13;
  auto const B = F * transpose(F);
  auto const devB = deviatoric_part(B);
  sigma = 0.5 * K * (J - Jinv) + (G * Jm53) * devB;
  Keff = 0.5 * K * (J + Jinv);
  Geff = G;
  potential = 0.5*G*(Jm23*trace(B) - 3.0) + 0.5*K*(0.5*(J*J - 1.0) - log(J));
}

HPC_ALWAYS_INLINE HPC_DEVICE void variational_J2_point(hpc::deformation_gradient<double> const &F, j2::Properties const props,
    hpc::time<double> const dt, hpc::stress<double> &sigma, hpc::pressure<double> &Keff, hpc::pressure<double>& Geff,
    hpc::energy_density<double> &potential, hpc::deformation_gradient<double> &Fp, hpc::strain<double> &eqps)
{
  auto const J = determinant(F);
  auto const Jm13 = 1.0 / std::cbrt(J);
  auto const Jm23 = Jm13 * Jm13;
  double const logJ = std::log(J);

  double const& K = props.K;
  double const& G = props.G;

  auto const We_vol = 0.5*K*logJ*logJ;
  auto const p = K*logJ/J;

  auto Fe_tr = F*hpc::inverse(Fp);
  auto dev_Ce_tr = Jm23*hpc::transpose(Fe_tr)*Fe_tr;
  auto dev_Ee_tr = 0.5*hpc::log(dev_Ce_tr);
  auto const dev_M_tr = 2.0*G*dev_Ee_tr;
  double const sigma_tr_eff = std::sqrt(1.5)*hpc::norm(dev_M_tr);
  auto Np = hpc::matrix3x3<double>::zero();
  if (sigma_tr_eff > 0) {
    Np = 1.5*dev_M_tr/sigma_tr_eff;
  }

  double S = j2::FlowStrength(props, eqps);
  double const r0 = sigma_tr_eff - S;
  auto r = r0;

  double delta_eqps = 0;
  double const tolerance = 1e-10;
  if (r > tolerance) {
    constexpr int max_iters = 8;
    int iters = 0;
    bool converged = false;
    while (!converged) {
      if (iters == max_iters) break;
      double H = j2::HardeningRate(props, eqps + delta_eqps) + j2::ViscoplasticHardeningRate(props, delta_eqps, dt);
      double dr = -3.0*G - H;
      double corr = -r/dr;
      delta_eqps += corr;
      double S = j2::FlowStrength(props, eqps + delta_eqps) + j2::ViscoplasticStress(props, delta_eqps, dt);
      r = sigma_tr_eff - 3.0*G*delta_eqps - S;
      converged = std::abs(r/r0) < tolerance;
      ++iters;
    }
    if (!converged) {
      std::cerr << "variational J2 diverged" << std::endl;
      throw;
      // TODO: handle non-convergence error
    }
    //std::cout << "variational J2 converged in " << iters << " iterations" << std::endl;
    auto dFp = hpc::exp(delta_eqps*Np);
    Fp = dFp*Fp;
    eqps += delta_eqps;
  }
  auto Ee_correction = delta_eqps*Np;
  auto dev_Ee = dev_Ee_tr - Ee_correction;
  auto dev_sigma = 1.0/J*hpc::transpose(hpc::inverse(Fe_tr))*(dev_M_tr - 2.0*G*Ee_correction)*hpc::transpose(Fe_tr);

  auto We_dev = G*hpc::inner_product(dev_Ee, dev_Ee);
  auto psi_star = j2::ViscoplasticDualKineticPotential(props, delta_eqps, dt);
  auto Wp = j2::HardeningPotential(props, eqps);

  sigma = dev_sigma+ p*hpc::matrix3x3<double>::identity();

  Keff = K;
  Geff = G;
  potential = We_vol + We_dev + Wp + psi_star;
}

}

