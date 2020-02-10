#ifndef J2_HARDENING_HPP_
#define J2_HARDENING_HPP_


namespace lgr {

namespace j2 {

struct Properties {
  double S0;
  double n;
  double eps0;
};


double
HardeningPotential(Properties props, double eqps);

double
FlowStrength(Properties props, double eqps);

double
HardeningRate(Properties props, double eqps);


} // namespace j2
} // namespace lgr


#endif /* J2_HARDENING_HPP_ */
