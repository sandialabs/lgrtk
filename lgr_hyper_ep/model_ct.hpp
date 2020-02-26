#ifndef LGR_HYPER_EP_MODEL_CT_HPP
#define LGR_HYPER_EP_MODEL_CT_HPP

#include <string>
#include <limits>
#include <algorithm>
#include <iostream>  // DEBUGGING

#include <hpc_symmetric3x3.hpp>

#include "defs.hpp"
#include "common.hpp"
#include "elastic.hpp"
#include "flow_stress.hpp"
#include "scalar_damage.hpp"

namespace lgr {
namespace hyper_ep {

/* Computes the radial return
 *
 * Yield function:
 *   S:S - Sqrt[2/3] * Y = 0
 * where S is the stress deviator.
 *
 * Equivalent plastic strain:
 *   ep = Integrate[Sqrt[2/3]*Sqrt[epdot:epdot], 0, t]
 *
 */
HPC_NOINLINE inline
ErrorCode
radial_return(Properties const props, hpc::symmetric_stress<double> const Te,
    hpc::deformation_gradient<double> const F, double const temp, double const dtime, hpc::symmetric_stress<double>& T,
    hpc::deformation_gradient<double>& Fp, double& ep, double& epdot, double& dp, StateFlag& flag)
{
  constexpr double tol1 = 1e-12;
  auto const tol2 = std::min(dtime, 1e-6);
  constexpr double twothird = 2.0 / 3.0;
  auto const sq2 = std::sqrt(2.0);
  auto const sq3 = std::sqrt(3.0);
  auto const sq23 = sq2 / sq3;
  auto const sq32 = 1.0 / sq23;
  auto const E = props.E;
  auto const nu = props.nu;
  auto const mu = E / 2.0 / (1.0 + nu);
  auto const twomu = 2.0 * mu;
  auto gamma = epdot * dtime * sq32;
  // Possible states at this point are TRIAL or REMAPPED
  if (flag != StateFlag::REMAPPED) flag = StateFlag::TRIAL;
  // check yield
  auto Y = flow_stress<props_hardening, props_rate_dep>(props, temp, ep, epdot, dp);
  auto const S0 = deviatoric_part(Te);
  auto const norm_S0 = norm(S0);
  auto f = norm_S0 / sq2 - Y / sq3;
  if (f <= tol1) {
    // Elastic loading
    T = 1. * Te;
    if (flag != StateFlag::REMAPPED) flag = StateFlag::ELASTIC;
  } else {
    int conv = 0;
    if (flag != StateFlag::REMAPPED) flag = StateFlag::PLASTIC;
    auto const N = S0 / norm_S0;  // Flow direction
    for (int iter = 0; iter < 100; ++iter) {
      // Compute the yield stress
      Y = flow_stress<props_hardening, props_rate_dep>(props, temp, ep, epdot, dp);
      // Compute g
      auto const g = norm_S0 - sq23 * Y - twomu * gamma;
      // Compute derivatives of g
      auto const dydg = dflow_stress<props_hardening, props_rate_dep>(props, temp, ep, epdot, dtime, dp);
      auto const dg = -twothird * dydg - twomu;
      // Update dgamma
      auto const dgamma = -g / dg;
      gamma += dgamma;
      // Update state
      auto const dep = std::max(sq23 * gamma, 0.0);
      epdot = dep / dtime;
      ep += dep;
      auto const S = S0 - twomu * gamma * N;
      f = norm(S) / sq2 - Y / sq3;
      if (f < tol1) {
        conv = 1;
        break;
      } else if (std::abs(dgamma) < tol2) {
        conv = 1;
        break;
      } else if (iter > 24 && f <= tol1 * 1000.0) {
        // Weaker convergence
        conv = 2;
        break;
      }
#ifdef LGR_HYPER_EP_VERBOSE_DEBUG
      std::cout << "Iteration: " << iter + 1 << "\n"
                << "\tROOTJ20: " << hpc::norm(S0) << "\n"
                << "\tROOTJ2: " << hpc::norm(S) << "\n"
                << "\tep: " << ep << "\n"
                << "\tepdot: " << epdot << "\n"
                << "\tgamma: " << gamma << "\n"
                << "\tg: " << g << "\n"
                << "\tdg: " << dg << "\n\n\n";
#endif
    }
    // Update the stress tensor
    T = Te - twomu * gamma * N;
    if (!conv) {
      return ErrorCode::RADIAL_RETURN_FAILURE;
    } else if (conv == 2) {
      // print warning about weaker convergence
    }
    // Update damage
    dp = scalar_damage<props_damage>(props, T, dp, temp, ep, epdot, dtime);
  }

  if (flag != StateFlag::ELASTIC) {
    // determine elastic deformation
    auto const jac = determinant(F);
    auto const Bbe = find_bbe(T, mu);
    auto const j13 = std::cbrt(jac);
    auto const j23 = j13 * j13;
    auto const Be = Bbe * j23;
    auto const Ve = sqrt_spd(Be);
    Fp = inverse(Ve) * F;
    if (flag == StateFlag::REMAPPED) {
      // Correct pressure term
      auto p = trace(T);
      auto const D1 = 6.0 * (1.0 - 2.0 * nu) / E;
      p = (2.0 * jac / D1 * (jac - 1.0)) - (p / 3.0);
      for (int i = 0; i < 3; ++i) T(i, i) = p;
    }
  }
  return ErrorCode::SUCCESS;
}

HPC_NOINLINE inline
ErrorCode
update(Properties const props, hpc::deformation_gradient<double> const F,
    double const dtime, double const temp, hpc::symmetric_stress<double>& T,
    hpc::deformation_gradient<double>& Fp, double& ep, double& epdot, double& dp, int& localized)
{
  auto const jac = determinant(F);

  // Determine the stress predictor.
  hpc::symmetric_stress<double> Te;
  auto const Fe = F * 1.;  //  inverse(Fp);
  ErrorCode err_c = ErrorCode::NOT_SET;
  Te = elastic_stress<props_elastic>(props, Fe, jac);

  // check yield and perform radial return (if applicable)
  auto flag = StateFlag::TRIAL;
  err_c = radial_return(props, Te, F, temp, dtime, T, Fp, ep, epdot, dp, flag);
  if (err_c != ErrorCode::SUCCESS) {
    return err_c;
  }

  bool is_localized = false;
  auto p = -trace(T) / 3.;
  auto const I = hpc::symmetric_stress<double>::identity();
  if (props.damage != Damage::NONE) {
    // If the particle has already failed, apply various erosion algorithms
    if (localized > 0) {
      if (props.allow_no_tension) {
        if (p < 0.0) {
          T = 0.0 * I;
        } else {
          T = -p * I;
        }
      }
      else if (props.allow_no_shear) {
        T = -p * I;
      }
      else if (props.set_stress_to_zero) {
        T = 0.0 * I;
      }
    }

    // Update damage and check modified TEPLA rule
    dp = scalar_damage<props_damage>(props, T, dp, temp, ep, epdot, dtime);
    double const por = 0.0;
    double const por_crit = 1.0;
    double const por_ratio = por / por_crit;
    double const D0dpDC = (props.D0 + dp) / props.DC;
    auto const tepla = por_ratio * por_ratio + D0dpDC * D0dpDC;
    if (tepla > 1.0) {
      is_localized = true;
    }
  }

  if (is_localized) {
    // If the localized material point fails again, set the stress to zero
    if (localized > 0) {
      dp = 0.0;
      T = 0.0 * I;
    } else {
      // set the particle localization flag to true
      localized = 1;
      dp = 0.0;
      // Apply various erosion algorithms
      if (props.allow_no_tension) {
        if (p < 0.0) {
          T = 0.0 * I;
        } else {
          T = -p * I;
        }
      }
      else if (props.allow_no_shear) {
        T = -p * I;
      }
      else if (props.set_stress_to_zero) {
        T = 0.0 * I;
      }
    }
  }
  return ErrorCode::SUCCESS;
}

}  // namespace hyper_ep
}  // namespace lgr

#endif  // LGR_HYPER_EP_MODEL_CT_HPP
