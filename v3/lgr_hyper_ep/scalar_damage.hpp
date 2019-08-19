#ifndef LGR_HYPER_EP_SCALAR_DAMAGE_HPP
#define LGR_HYPER_EP_SCALAR_DAMAGE_HPP

#include "common.hpp"
#include <hpc_symmetric3x3.hpp>

namespace lgr {
namespace hyper_ep {
namespace impl {

template <Damage damage>
HPC_NOINLINE inline double
scalar_damage(
  Properties const /* props */, hpc::symmetric_stress<double>& /* T */,
  double const /* dp */, double const /* temp */, double const /* ep */,
  double const /* epdot */, double const /* dtime */)
{
  std::cout << "Must provide partial specialization\n";
  assert (false);
}

template<>
HPC_NOINLINE inline double
scalar_damage<Damage::NONE>(
  Properties const /* props */, hpc::symmetric_stress<double>& /* T */,
  double const /* dp */, double const /* temp */, double const /* ep */,
  double const /* epdot */, double const /* dtime */)
{
  return 0.0;
}

template<>
HPC_NOINLINE inline double
scalar_damage<Damage::JOHNSON_COOK>(
  Properties const props, hpc::symmetric_stress<double>& T,
  double const dp, double const temp, double const /* ep */,
  double const epdot, double const dtime)
{
  double const tolerance = 1.0e-10;
  auto const I = hpc::symmetric_stress<double>::identity();
  auto const T_mean = (trace(T) / 3.0);
  auto const S = T - I * T_mean;
  auto const norm_S = norm(S);
  auto const S_eq = std::sqrt(norm_S * norm_S * 1.5);

  double eps_f = props.eps_f_min;
  double sig_star = (std::abs(S_eq) > 1e-16) ? T_mean / S_eq : 0.0;
  if (sig_star < 1.5) {
    // NOT SPALL
    // sig_star < 1.5 indicates spall conditions are *not* met and the failure
    // strain must be calculated.
    sig_star = std::max(std::min(sig_star, 1.5), -1.5);

    // Stress contribution to damage
    double stress_contrib = props.D1 + props.D2 * exp(props.D3 * sig_star);

    // Strain rate contribution to damage
    double dep_contrib = 1.0;
    if (epdot < 1.0) {
      dep_contrib = std::pow((1.0 + epdot), props.D4);
    } else {
      dep_contrib = 1.0 + props.D4 * std::log(epdot);
    }

    double temp_contrib = 1.0;
    auto const temp_ref = props.D6;
    auto const temp_melt = props.D7;
    if (std::abs(temp_melt-std::numeric_limits<double>::max())+1.0 != 1.0) {
      auto const tstar = temp > temp_melt ? 1.0 : (temp - temp_ref) / (temp_melt - temp_ref);
      temp_contrib += props.D5 * tstar;
    }

    // Calculate the updated scalar damage parameter
    eps_f = stress_contrib * dep_contrib * temp_contrib;
  }

  if (eps_f < tolerance) return dp;

  // Calculate plastic strain increment
  auto const dep = epdot * dtime;
  auto const ddp = dep / eps_f;
  return (dp + ddp < tolerance) ? 0.0 : dp + ddp;
}

} // namespace impl

template <Damage damage>
HPC_NOINLINE inline double
scalar_damage(
  Properties const props, hpc::symmetric_stress<double>& T,
  double const dp, double const temp, double const ep,
  double const epdot, double const dtime)
{
  return impl::scalar_damage<damage>(props, T, dp, temp, ep, epdot, dtime);
}

} // namespace hyper_ep
} // namespace lgr

#endif // LGR_HYPER_EP_SCALAR_DAMAGE_HPP
