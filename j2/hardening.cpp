#include "hardening.hpp"
#include <cmath>

namespace lgr {

namespace j2 {

double
HardeningPotential(Properties const props, double const eqps)
{
  double const& S0 = props.S0;
  double const& n = props.n;
  double const& eps0 = props.eps0;

  double const exponent = (1.0 + n)/n;

  return S0*eps0/exponent*std::pow(1.0 + eqps/eps0, exponent);
}

double
FlowStrength(Properties const props, double const eqps)
{
  double const& S0 = props.S0;
  double const& n = props.n;
  double const& eps0 = props.eps0;

  return S0*std::pow(1.0 + eqps/eps0, 1.0/n);
}

double
HardeningRate(Properties const props, double const eqps)
{
  double const& S0 = props.S0;
  double const& n = props.n;
  double const& eps0 = props.eps0;

  return S0/(eps0*n)*std::pow(1.0 + eqps/eps0, (1.0 - n)/n);
}

double
ViscoplasticDualKineticPotential(Properties const props, double const delta_eqps, double const dt)
{
  double const& Svis0 = props.Svis0;
  double const& m = props.m;
  double const& eps_dot0 = props.eps_dot0;

  double const exponent = (1.0 + m)/m;
  double const eqps_dot = delta_eqps/dt;

  return dt*Svis0*eps_dot0/exponent*std::pow(1.0 + eqps_dot/eps_dot0, exponent);
}

double
ViscoplasticStress(Properties const props, double const delta_eqps, double const dt)
{
  double const& Svis0 = props.Svis0;
  double const& m = props.m;
  double const& eps_dot0 = props.eps_dot0;

  double const eqps_dot = delta_eqps/dt;

  return Svis0*std::pow(1.0 + eqps_dot/eps_dot0, 1.0/m);
}

double
ViscoplasticHardeningRate(Properties const props, double const delta_eqps, double const dt)
{
  double const& Svis0 = props.Svis0;
  double const& m = props.m;
  double const& eps_dot0 = props.eps_dot0;

  double const eqps_dot = delta_eqps/dt;

  return Svis0/(eps_dot0*m*dt)*std::pow(1.0 + eqps_dot/eps_dot0, (1.0 - m)/m);
}

} // namespace j2
} // namespace lgr



