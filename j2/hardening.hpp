#pragma once

#include <hpc_limits.hpp>
#include <hpc_macros.hpp>
#include <hpc_math.hpp>

namespace lgr {

namespace j2 {

struct Properties
{
  double K;
  double G;

  double Y0;
  double n;
  double eps0;

  double Svis0;
  double m;
  double eps_dot0;
};

HPC_ALWAYS_INLINE HPC_HOST_DEVICE double
HardeningPotential(Properties const props, double const eqps)
{
  double const& Y0   = props.Y0;
  double const& n    = props.n;
  double const& eps0 = props.eps0;

  if (n == hpc::numeric_limits<double>::infinity()) return Y0 * eqps;

  double const exponent = (1.0 + n) / n;

  return Y0 * eps0 / exponent * (std::pow(1.0 + eqps / eps0, exponent) - 1.0);
}

HPC_ALWAYS_INLINE HPC_HOST_DEVICE double
FlowStrength(Properties const props, double const eqps)
{
  double const& Y0   = props.Y0;
  double const& n    = props.n;
  double const& eps0 = props.eps0;

  if (n == hpc::numeric_limits<double>::infinity()) return Y0;

  return Y0 * std::pow(1.0 + eqps / eps0, 1.0 / n);
}

HPC_ALWAYS_INLINE HPC_HOST_DEVICE double
HardeningRate(Properties const props, double const eqps)
{
  double const& Y0   = props.Y0;
  double const& n    = props.n;
  double const& eps0 = props.eps0;

  if (n == hpc::numeric_limits<double>::infinity()) return 0.0;

  return Y0 / (eps0 * n) * std::pow(1.0 + eqps / eps0, (1.0 - n) / n);
}

HPC_ALWAYS_INLINE HPC_HOST_DEVICE double
ViscoplasticDualKineticPotential(Properties const props, double const delta_eqps, double const dt)
{
  double const& Svis0    = props.Svis0;
  double const& m        = props.m;
  double const& eps_dot0 = props.eps_dot0;

  if (Svis0 == 0.0) return 0.0;

  double const exponent = (1.0 + m) / m;
  double       psi_star = 0.0;
  if (delta_eqps > 0) {
    psi_star = dt * Svis0 * eps_dot0 / exponent * std::pow(delta_eqps / dt / eps_dot0, exponent);
  }
  return psi_star;
}

HPC_ALWAYS_INLINE HPC_HOST_DEVICE double
ViscoplasticStress(Properties const props, double const delta_eqps, double const dt)
{
  double const& Svis0    = props.Svis0;
  double const& m        = props.m;
  double const& eps_dot0 = props.eps_dot0;

  if (Svis0 == 0.0) return 0.0;

  double Svis = 0;
  if (delta_eqps > 0) {
    Svis = Svis0 * std::pow(delta_eqps / dt / eps_dot0, 1.0 / m);
  }

  return Svis;
}

HPC_ALWAYS_INLINE HPC_HOST_DEVICE double
ViscoplasticHardeningRate(Properties const props, double const delta_eqps, double const dt)
{
  double const& Svis0    = props.Svis0;
  double const& m        = props.m;
  double const& eps_dot0 = props.eps_dot0;

  if (Svis0 == 0.0) return 0.0;

  double Hvis = 0;
  if (delta_eqps > 0) {
    Hvis = Svis0 / (eps_dot0 * m * dt) * std::pow(delta_eqps / dt / eps_dot0, (1.0 - m) / m);
  }

  return Hvis;
}

}  // namespace j2
}  // namespace lgr
