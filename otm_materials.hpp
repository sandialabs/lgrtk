#pragma once

#include <hpc_dimensional.hpp>
#include <hpc_macros.hpp>
#include <hpc_math.hpp>
#include <hpc_matrix3x3.hpp>
#include <hpc_symmetric3x3.hpp>
#include <iostream>
#include <j2/hardening.hpp>

namespace lgr {

HPC_ALWAYS_INLINE HPC_HOST_DEVICE void
neo_Hookean_point(
    hpc::deformation_gradient<double> const& F,
    hpc::pressure<double> const              K,
    hpc::pressure<double> const              G,
    hpc::stress<double>&                     sigma,
    hpc::pressure<double>&                   Keff,
    hpc::pressure<double>&                   Geff,
    hpc::energy_density<double>&             potential)
{
  auto const J    = determinant(F);
  auto const Jinv = 1.0 / J;
  auto const Jm13 = 1.0 / std::cbrt(J);
  auto const Jm23 = Jm13 * Jm13;
  auto const Jm53 = (Jm23 * Jm23) * Jm13;
  auto const B    = F * transpose(F);
  auto const devB = deviatoric_part(B);
  sigma           = 0.5 * K * (J - Jinv) + (G * Jm53) * devB;
  Keff            = 0.5 * K * (J + Jinv);
  Geff            = G;
  potential       = 0.5 * G * (Jm23 * trace(B) - 3.0) + 0.5 * K * (0.5 * (J * J - 1.0) - log(J));
}

HPC_ALWAYS_INLINE HPC_HOST_DEVICE void
variational_J2_point(
    hpc::deformation_gradient<double> const& F,
    j2::Properties const                     props,
    hpc::time<double> const                  dt,
    hpc::stress<double>&                     sigma,
    hpc::pressure<double>&                   Keff,
    hpc::pressure<double>&                   Geff,
    hpc::energy_density<double>&             potential,
    hpc::deformation_gradient<double>&       Fp,
    hpc::strain<double>&                     eqps)
{
  auto const J    = determinant(F);
  auto const Jm13 = 1.0 / std::cbrt(J);
  auto const Jm23 = Jm13 * Jm13;
  auto const logJ = std::log(J);

  auto const& K = props.K;
  auto const& G = props.G;

  auto const We_vol = 0.5 * K * logJ * logJ;
  auto const p      = K * logJ / J;

  auto       Fe_tr        = F * hpc::inverse(Fp);
  auto       dev_Ce_tr    = Jm23 * hpc::transpose(Fe_tr) * Fe_tr;
  auto       dev_Ee_tr    = 0.5 * hpc::log(dev_Ce_tr);
  auto const dev_M_tr     = 2.0 * G * dev_Ee_tr;
  auto const sigma_tr_eff = std::sqrt(1.5) * hpc::norm(dev_M_tr);
  auto       Np           = hpc::matrix3x3<double>::zero();
  if (sigma_tr_eff > 0) {
    Np = 1.5 * dev_M_tr / sigma_tr_eff;
  }

  auto       S0 = j2::FlowStrength(props, eqps);
  auto const r0 = sigma_tr_eff - S0;
  auto       r  = r0;

  auto       delta_eqps         = 0.0;
  auto const residual_tolerance = 1e-10;
  auto const deqps_tolerance    = 1e-10;
  if (r > residual_tolerance) {
    constexpr auto max_iters = 8;
    auto           iters     = 0;
    auto           merit_old = 1.0;
    auto           merit_new = 1.0;

    auto converged = false;
    while (!converged) {
      if (iters == max_iters) break;
      auto ls_is_finished = false;
      auto delta_eqps0    = delta_eqps;
      merit_old           = r * r;
      auto H  = j2::HardeningRate(props, eqps + delta_eqps) + j2::ViscoplasticHardeningRate(props, delta_eqps, dt);
      auto dr = -3.0 * G - H;
      auto correction = -r / dr;

      // line search
      auto       alpha                  = 1.0;
      auto       line_search_iterations = 0;
      auto const backtrack_factor       = 0.1;
      auto const decrease_factor        = 1e-5;
      while (!ls_is_finished) {
        if (line_search_iterations == 20) {
          // line search has failed to satisfactorily improve newton step
          // just take the full newton step and hope for the best
          alpha = 1;
          break;
        }
        ++line_search_iterations;
        delta_eqps = delta_eqps0 + alpha * correction;
        if (delta_eqps < 0) delta_eqps = 0;
        auto Yeq          = j2::FlowStrength(props, eqps + delta_eqps);
        auto Yvis         = j2::ViscoplasticStress(props, delta_eqps, dt);
        auto residual     = sigma_tr_eff - 3.0 * G * delta_eqps - (Yeq + Yvis);
        merit_new         = residual * residual;
        auto decrease_tol = 1.0 - 2.0 * alpha * decrease_factor;
        if (merit_new <= decrease_tol * merit_old) {
          merit_old      = merit_new;
          ls_is_finished = true;
        } else {
          auto alpha_old = alpha;
          alpha          = alpha_old * alpha_old * merit_old / (merit_new - merit_old + 2.0 * alpha_old * merit_old);
          if (backtrack_factor * alpha_old > alpha) {
            alpha = backtrack_factor * alpha_old;
          }
        }
      }
      auto S    = j2::FlowStrength(props, eqps + delta_eqps) + j2::ViscoplasticStress(props, delta_eqps, dt);
      r         = sigma_tr_eff - 3.0 * G * delta_eqps - S;
      converged = (std::abs(r / r0) < residual_tolerance) || (delta_eqps < deqps_tolerance);
      ++iters;
    }
    if (!converged) {
      HPC_DUMP("variational J2 did not converge to specified tolerance 1.0e-10\n");
      // TODO: handle non-convergence error
    }
    // std::cout << "variational J2 converged in " << iters << " iterations" <<
    // std::endl;
    auto dFp = hpc::exp(delta_eqps * Np);
    Fp       = dFp * Fp;
    eqps += delta_eqps;
  }
  auto Ee_correction = delta_eqps * Np;
  auto dev_Ee        = dev_Ee_tr - Ee_correction;
  auto dev_sigma =
      1.0 / J * hpc::transpose(hpc::inverse(Fe_tr)) * (dev_M_tr - 2.0 * G * Ee_correction) * hpc::transpose(Fe_tr);

  auto We_dev   = G * hpc::inner_product(dev_Ee, dev_Ee);
  auto psi_star = j2::ViscoplasticDualKineticPotential(props, delta_eqps, dt);
  auto Wp       = j2::HardeningPotential(props, eqps);

  sigma = dev_sigma + p * hpc::matrix3x3<double>::identity();

  Keff      = K;
  Geff      = G;
  potential = We_vol + We_dev + Wp + psi_star;
}

// Mie–Grüneisen EOS. Adapted from LGR v2
//
// The locus of shocked states comprises a Hugoniot curve for the material.
// This implementation of the Mie–Grüneisen EOS is based on the Hugoniot
// relations. References to 'ph' and 'eh' are the pressure and energy on
// the Hugoniot, respectively.
//
// In many relatively stiff solid materials, notably metals, ceramics,
// and minerals, for example, the Hugoniot states at shock pressures
// exceeding the elastic limit are accurately represented by a linear
// relationship between shock velocity (u_s) and particle velocity (u_p):
//
//                     u_s = c_0 + s * u_p
//
// Then, by using the Hugoniot equations for the conservation of mass
// and momentum we get to:
//
//                             c_0^2
//   p_h = rho_0 * ------------------------------ * (1 - rho_0 / rho)
//                  (1 - s * (1 - rho_0 / rho))^2
//
// This assumes that the reference pressure is zero and the reference
// density is rho_0.
//
//            HUGONIOT CONSERVATION LAWS
//            --------------------------
//
// Conservation of mass:
// (1)  rho1 * us = rho2 * (us - u2)
// (2)         u2 = us * (1- rho1 / rho2)
//
// Conservation of momentum:
// (3)  p2 - p1 = rho1 * us * u2
//
// Conservation of energy:
// (4)  e2 - e1 = (p2 + p1) * (1 / rho1 - 1 / rho2) / 2
//
// Merging conservation of mass and momentum:
// (5)  p2 - p1 = rho1 * us^2 * (1 - rho1 / rho2)
//
//                    ASSUMPTIONS
//                    -----------
//
// Linear us-up relation - simplified using Equation (2):
// (6)  us = c0 + s * up = c0 + s * u2
// (7)  us = c0 / (1 - s * (1 - rho1 / rho2))
//
// The following substitutions are made to get to the final form:
//   * p2 = ph
//   * p1 = 0
//   * e2 = eh
//   * e1 = 0
//   * rho2 = rho
//   * rho1 = rho0
//
//                   FINAL FORMS
//                   -----------
//
// Reference Hugoniot Pressure - Equation (3) plus Equations (7) and (2)
// (8)  ph = rho0 * us^2 * (1 - rho0 / rho)
// (9)  ph = rho0 * c0^2 / (1 - s * (1 - rho0 / rho))^2 * (1 - rho0 / rho)
//
// Reference Hugoniot Energy - Equation (4)
// (10)  eh = ph * (1 / rho0 - 1 / rho ) / 2
//
// Wrapping it all together, we integrate the Gruneisen model to get and
// explicitly note the dependence of 'ph' and 'eh' on 'rho':
// (11)  p - ph(rho) = Gamma * rho * (e - eh(rho))
// (12)  p = ph(rho) + Gamma * rho * (e - eh(rho))
//
HPC_ALWAYS_INLINE HPC_HOST_DEVICE void
mie_gruniesen_eos(
    hpc::density<double> const         reference_density,
    hpc::density<double> const         current_density,
    hpc::specific_energy<double> const internal_energy,
    hpc::adimensional<double> const    gamma,
    hpc::speed<double> const           reference_wave_speed,
    hpc::adimensional<double> const    s,
    hpc::pressure<double>&             pressure,
    hpc::speed<double>&                wave_speed)
{
  auto const mu = 1.0 - reference_density / current_density;
  // d mu / d rho
  auto const dmu = reference_density / (current_density * current_density);
  // limit 'us' to not drop below 'reference_wave_speed'
  auto const us = reference_wave_speed / (1.0 - s * std::max(0.0, mu));
  auto const ph = reference_density * us * us * mu;
  auto const eh = 0.5 * ph * mu / reference_density;
  // derivative of pressure with respect to density
  auto const in_compression = mu > 0.0;
  auto const dus            = in_compression == true ? (us * s / (1.0 - s * mu)) * dmu : 0.0;
  auto const dph            = 2.0 * reference_density * us * dus * mu + reference_density * us * us * dmu;
  auto const deh            = 0.5 * (mu * dph + ph * dmu) / reference_density;
  auto const dpdrho         = dph - gamma * reference_density * deh;
  // derivative of pressure with repect to energy
  auto const dpde = gamma * reference_density;
  // Pressure
  pressure = ph + gamma * reference_density * (internal_energy - eh);
  // Wave speed
  auto const bulk_modulus = current_density * dpdrho + (pressure / current_density) * dpde;
  wave_speed              = std::sqrt(bulk_modulus / current_density);
}

}  // namespace lgr
