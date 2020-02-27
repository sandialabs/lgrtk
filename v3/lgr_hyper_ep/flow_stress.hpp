#ifndef LGR_HYPER_EP_FLOW_STRESS_HPP
#define LGR_HYPER_EP_FLOW_STRESS_HPP

#include "common.hpp"

namespace lgr {
namespace hyper_ep {
namespace impl {

template <Hardening hardening, RateDependence rate_dependence>
HPC_NOINLINE inline double
flow_stress(
  Properties /* props */, double const /* temp */,
  double const /* ep */, double const /* epdot */)
{
  std::cout << "Must provide partial specialization\n";
  assert (false);
}

/// \brief Constant flow stress given by
//
//     Y = A
//
template <>
HPC_NOINLINE inline double
flow_stress<Hardening::NONE, RateDependence::NONE>(
  Properties props, double const /* temp */,
  double const /* ep */, double const /* epdot */)
{
  return props.A;
}

/// \brief Linearly hardening flow stress given by
//
//     Y = A + B ep
//
template <>
HPC_NOINLINE inline double
flow_stress<Hardening::LINEAR_ISOTROPIC, RateDependence::NONE>(
  Properties props, double const /* temp */,
  double const ep, double const /* epdot */)
{
  return props.A + props.B * ep;
}

/// \brief Power law hardening flow stress given by
//
//     Y = A + B ep^n
//
template <>
HPC_NOINLINE inline double
flow_stress<Hardening::POWER_LAW, RateDependence::NONE>(
  Properties props, double const /* temp */,
  double const ep, double const /* epdot */)
{
  auto Y = props.A;
  if (props.B > 0.0)
    Y += (std::abs(props.n) > 0.0) ? props.B * std::pow(ep, props.n) : props.B;
  return Y;
}

/// \brief Rate-independent Zerilli-Armstrong flow stress given by
//
//     Y = (A + B ep^n) + (C1 + C2 Sqrt[ep]) Exp[-C3 T]
//
template <>
HPC_NOINLINE inline double
flow_stress<Hardening::ZERILLI_ARMSTRONG, RateDependence::NONE>(
  Properties props, double const temp, double const ep, double const epdot)
{
  auto Y = flow_stress<Hardening::POWER_LAW, RateDependence::NONE>(props, temp, ep, epdot);
  Y += (props.C1 + props.C2 * std::sqrt(ep)) * std::exp(-props.C3 * temp);
  return Y;
}

/// \brief Rate-dependent Zerilli-Armstrong flow stress given by
//
//     Y = (A + B ep^n) + (C1 + C2 Sqrt[ep]) Exp[-C3 T] Exp[C4 Log[epdot] T]
//
template <>
HPC_NOINLINE inline double
flow_stress<Hardening::ZERILLI_ARMSTRONG, RateDependence::ZERILLI_ARMSTRONG>(
  Properties props, double const temp, double const ep, double const epdot)
{
  auto Y = flow_stress<Hardening::ZERILLI_ARMSTRONG, RateDependence::NONE>(props, temp, ep, epdot);
  Y *= std::exp(props.C4 * std::log(epdot) * temp);
  return Y;
}

/// \brief Rate-independent Johnson-Cook flow stress given by
//
//     Y = (A + B * ep^n) (1 - tstar^m)
//
template <>
HPC_NOINLINE inline double
flow_stress<Hardening::JOHNSON_COOK, RateDependence::NONE>(
  Properties props, double const temp, double const ep, double const epdot)
{
  // Constant contribution
  auto Y = flow_stress<Hardening::POWER_LAW, RateDependence::NONE>(props, temp, ep, epdot);
  // Temperature contribution
  if (std::abs(props.C2 - std::numeric_limits<double>::max()) + 1.0 != 1.0) {
    auto const tstar = (temp > props.C2) ? 1.0 : ((temp - props.C1) / (props.C2 - props.C1));
    Y *= (tstar < 0.0) ? (1.0 - tstar) : (1.0 - std::pow(tstar, props.C3));
  }
  return Y;
}

/// \brief Rate-dependent Johnson-Cook flow stress given by
//
//     Y = (A + B * ep^n) (1 - tstar^m) (1 + C Log[epdot/epdot0])
//
template <>
HPC_NOINLINE inline double
flow_stress<Hardening::JOHNSON_COOK, RateDependence::JOHNSON_COOK>(
  Properties props, double const temp, double const ep, double const epdot)
{
  auto Y = flow_stress<Hardening::JOHNSON_COOK, RateDependence::NONE>(props, temp, ep, epdot);
  auto const rfac = epdot / props.eps_dot0;
  // FIXME: This assumes that all the strain rate is plastic. Should use actual
  // strain rate.
  // Rate of plastic strain contribution
  if (props.C4 > 0.0) {
    Y *= (rfac < 1.0) ? std::pow((1.0 + rfac), props.C4) : (1.0 + props.C4 * std::log(rfac));
  }
  return Y;
}

/// \brief non-specialized implementation of derivative of flow stress with
/// respect to plastic strain
template <Hardening hardening, RateDependence rate_dependence>
HPC_NOINLINE inline double
dflow_stress(
  Properties const /* props */, double const /* temp */, double const /* ep */,
  double const /* epdot */, double const /* dtime */)
{
  std::cout << "Must provide partial specialization\n";
  assert (false);
}

/// \brief Derivative of constant flow stress with respect to plastic strain, given by
//
//     dY = 0
//
template <>
HPC_NOINLINE inline double
dflow_stress<Hardening::NONE, RateDependence::NONE>(
  Properties const /* props */, double const /* temp */, double const /* ep */,
  double const /* epdot */, double const /* dtime */)
{
  return 0.0;
}

/// \brief Derivative of isotropic hardening flow stress with respect
//    to plastic strain, given by
//
//     dY = B
//
template <>
HPC_NOINLINE inline double
dflow_stress<Hardening::LINEAR_ISOTROPIC, RateDependence::NONE>(
  Properties const props, double const /* temp */, double const /* ep */,
  double const /* epdot */, double const /* dtime */)
{
  return props.B;
}

/// \brief Derivative of power law hardening flow stress with respect
//    to plastic strain, given by
//
//     dY = n B * ep^(n - 1)
//
template <>
HPC_NOINLINE inline double
dflow_stress<Hardening::POWER_LAW, RateDependence::NONE>(
  Properties const props, double const /* temp */, double const ep,
  double const /* epdot */, double const /* dtime */)
{
  return (ep > 0.0) ? props.B * props.n * std::pow(ep, props.n - 1) : 0.0;
}

/// \brief Derivative of rate-independent Zerilli-Armstrong flow stress with respect
//    to plastic strain, given by
//
//     dY = n B ep^(n - 1) + C2 / 2 / Sqrt[ep] Exp[-C3 T]
//
template <>
HPC_NOINLINE inline double
dflow_stress<Hardening::ZERILLI_ARMSTRONG, RateDependence::NONE>(
  Properties const props, double const temp, double const ep,
  double const epdot, double const dtime)
{
  auto dY = dflow_stress<Hardening::POWER_LAW, RateDependence::NONE>(props, temp, ep, epdot, dtime);
  dY += .5 * props.C2 / ((ep <= 0.0) ? 1.0e-12 : std::sqrt(ep)) * std::exp(-props.C3 * temp);
  return dY;
}

/// \brief Derivative of rate-independent Zerilli-Armstrong flow stress with respect
//    to plastic strain, given by
//
//     Y = (A + B ep^n) + (C1 + C2 Sqrt[ep]) Exp[-C3 T] Exp[C4 Log[epdot] T]
//
template <>
HPC_NOINLINE inline double
dflow_stress<Hardening::ZERILLI_ARMSTRONG, RateDependence::ZERILLI_ARMSTRONG>(
  Properties const props, double const temp, double const ep,
  double const epdot, double const dtime)
{
  auto dY = dflow_stress<Hardening::POWER_LAW, RateDependence::NONE>(props, temp, ep, epdot, dtime);
  auto alpha = props.C3;
  alpha -= props.C4 * std::log(epdot);
  dY += .5 * props.C2 / std::sqrt(ep <= 0.0 ? 1.e-8 : ep) * std::exp(-alpha * temp);

  auto const term1 = props.C1 * props.C4 * temp * std::exp(-alpha * temp);
  auto const term2 = props.C2 * sqrt(ep) * props.C4 * temp * std::exp(-alpha * temp);
  dY += (term1 + term2) / (epdot <= 0.0 ? 1.e-8 : epdot) / dtime;
  return dY;
}

/// \brief Derivative of rate-independent Johnson-Cook flow stress with respect
//    to plastic strain, given by
//
//     dY = m B * ep^(m - 1) (1 - tstar^m)
//
template <>
HPC_NOINLINE inline double
dflow_stress<Hardening::JOHNSON_COOK, RateDependence::NONE>(
  Properties const props, double const temp, double const ep,
  double const epdot, double const dtime)
{
  auto dY = dflow_stress<Hardening::POWER_LAW, RateDependence::NONE>(props, temp, ep, epdot, dtime);
  // Calculate temperature contribution
  double temp_contrib = 1.0;
  if (std::abs(props.C2 - std::numeric_limits<double>::max()) + 1.0 != 1.0) {
    auto const tstar = (temp > props.C2) ? 1.0 : (temp - props.C1) / (props.C2 - props.C1);
    temp_contrib = (tstar < 0.0) ? (1.0 - tstar) : (1.0 - std::pow(tstar, props.C3));
  }
  dY *= temp_contrib;
  return dY;
}

/// \brief Derivative of rate-dependent Johnson-Cook flow stress with respect to
///   plastic strain, given by
//
//     dY = d(A + B * ep^m) (1 - tstar^m) (1 + C log(epdot/epdot0))
//        + (A + B * ep^m) (1 - tstar^m) d(1 + C log(epdot/epdot0))
//        = m B * ep^(m - 1)) (1 - tstar^m) (1 + C log(epdot/epdot0))
//        + C Y d(log(epdot/epdot0))
//        = m B * ep^(m - 1)) (1 - tstar^m) (1 + C log(epdot/epdot0))
//        + C Y (1 + epdot/epdot0)^(C - 1) / dt
//
template <>
HPC_NOINLINE inline double
dflow_stress<Hardening::JOHNSON_COOK, RateDependence::JOHNSON_COOK>(
  Properties const props, double const temp, double const ep,
  double const epdot, double const dtime)
{
  auto dY = dflow_stress<Hardening::JOHNSON_COOK, RateDependence::NONE>(props, temp, ep, epdot, dtime);
  auto const rfac = epdot / props.eps_dot0;

  // Calculate strain rate contribution
  auto const term1 = (rfac < 1.0) ? (std::pow((1.0 + rfac), props.C4)) : (1.0 + props.C4 * std::log(rfac));
  dY *= term1;

  auto term2 = flow_stress<Hardening::JOHNSON_COOK, RateDependence::NONE>(props, temp, ep, epdot);
  if (rfac < 1.0) {
    term2 *= props.C4 * std::pow((1.0 + rfac), (props.C4 - 1.0));
  } else {
    term2 *= props.C4 / rfac;
  }
  dY += term2 / dtime;
  return dY;
}

} // namespace impl

/// \brief Public interface to flow stress implementations
template <Hardening hardening, RateDependence rate_dependence>
HPC_NOINLINE inline double
flow_stress(
  Properties const props, double const temp, double const ep,
  double const epdot, double const dp)
{
  auto Y = impl::flow_stress<hardening, rate_dependence>(props, temp, ep, epdot);
  return (1. - dp) * Y;
}

template <Hardening hardening, RateDependence rate_dependence>
HPC_NOINLINE inline double
dflow_stress(
  Properties const props, double const temp, double const ep,
  double const epdot, double const dtime, double const dp)
{
  auto dY = impl::dflow_stress<hardening, rate_dependence>(props, temp, ep, epdot, dtime);
  constexpr double sq23 = 0.8164965809277261;
  return (1. - dp) * sq23 * dY;
}

} // namespace hyper_ep
} // namespace lgr

#endif // LGR_HYPER_EP_FLOW_STRESS_HPP
