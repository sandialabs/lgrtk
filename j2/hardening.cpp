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

  return S0*eps0/exponent*(std::pow(1.0 + eqps/eps0, exponent) - 1.0);
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
  double psi_star = 0;
  if (delta_eqps > 0) {
    psi_star = dt*Svis0*eps_dot0/exponent*std::pow(delta_eqps/dt/eps_dot0, exponent);
  }
  return psi_star;
}

double
ViscoplasticStress(Properties const props, double const delta_eqps, double const dt)
{
  double const& Svis0 = props.Svis0;
  double const& m = props.m;
  double const& eps_dot0 = props.eps_dot0;
  double Svis = 0;
  if (delta_eqps > 0) {
    Svis = Svis0*std::pow(delta_eqps/dt/eps_dot0, 1.0/m);
  }

  return Svis;
}

double
ViscoplasticHardeningRate(Properties const props, double const delta_eqps, double const dt)
{
  double const& Svis0 = props.Svis0;
  double const& m = props.m;
  double const& eps_dot0 = props.eps_dot0;
  double Hvis = 0;
  if (delta_eqps > 0) {
    Hvis = Svis0/(eps_dot0*m*dt)*std::pow(delta_eqps/dt/eps_dot0, (1.0 - m)/m);
  }

  return Hvis;
}

} // namespace j2
} // namespace lgr



