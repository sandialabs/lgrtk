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
FlowStrength(Properties props, double eqps)
{
  double const& S0 = props.S0;
  double const& n = props.n;
  double const& eps0 = props.eps0;

  return S0*std::pow(1.0 + eqps/eps0, 1.0/n);
}

double
HardeningRate(Properties props, double eqps)
{
  double const& S0 = props.S0;
  double const& n = props.n;
  double const& eps0 = props.eps0;

  return S0/(eps0*n)*std::pow(1.0 + eqps/eps0, (1.0 - n)/n);
}


} // namespace j2
} // namespace lgr



