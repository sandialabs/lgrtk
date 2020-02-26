#ifndef LGR_HYPER_EP_ELASTIC_HPP
#define LGR_HYPER_EP_ELASTIC_HPP

#include "common.hpp"
#include <hpc_symmetric3x3.hpp>

namespace lgr {
namespace hyper_ep {
namespace impl {

template <Elastic elastic>
HPC_NOINLINE inline
hpc::symmetric_stress<double>
elastic_stress(
  Properties const /* props */,
  hpc::deformation_gradient<double> const /* Fe */,
  double const /* jac */)
{
  std::cout << "Must provide partial specialization\n";
  assert (false);
}

template <>
HPC_NOINLINE inline
hpc::symmetric_stress<double>
elastic_stress<Elastic::LINEAR_ELASTIC>(
  Properties const props,
  hpc::deformation_gradient<double> const Fe,
  double const /* jac */)
{
  auto const E = props.E;
  auto const nu = props.nu;
  auto const K = E / (3.0 * (1.0 - 2.0 * nu));
  auto const G = E / 2.0 / (1.0 + nu);
  auto const grad_u = Fe - hpc::deformation_gradient<double>::identity();
  auto const strain = symmetric_part(grad_u);
  return (3.0 * K) * isotropic_part(strain) + (2.0 * G) * deviatoric_part(strain);
}

/*
 * Update the stress using Neo-Hookean hyperelasticity
 *
 */
template <>
HPC_NOINLINE inline
hpc::symmetric_stress<double>
elastic_stress<Elastic::NEO_HOOKEAN>(
  Properties const props,
  hpc::deformation_gradient<double> const Fe,
  double const jac)
{
  auto const E = props.E;
  auto const nu = props.nu;
  // Jacobian and distortion tensor
  auto const scale = 1.0 / std::cbrt(jac);
  auto const Fb = scale * Fe;
  // Elastic moduli
  auto const C10 = E / (4.0 * (1.0 + nu));
  auto const D1 = 6.0 * (1.0 - 2.0 * nu) / E;
  auto const EG = 2.0 * C10 / jac;
  // Deviatoric left Cauchy-Green deformation tensor
  auto Bb = self_times_transpose(Fb);
  // Deviatoric Cauchy stress
  auto const TRBb = trace(Bb) / 3.0;
  for (int i = 0; i < 3; ++i) Bb(i, i) -= TRBb;
  auto T = hpc::symmetric_stress<double>(EG * Bb);
  // Pressure response
  auto const PR = 2.0 / D1 * (jac - 1.0);
  for (int i = 0; i < 3; ++i) T(i, i) += PR;
  return T;
}

} // namespace impl

template <Elastic elastic>
HPC_NOINLINE inline
hpc::symmetric_stress<double>
elastic_stress(
  Properties const props,
  hpc::deformation_gradient<double> const Fe,
  double const jac)
{
  return impl::elastic_stress<elastic>(props, Fe, jac);
}

} // namespace hyper_ep
} // namespace lgr

#endif // LGR_HYPER_EP_ELASTIC_HPP
