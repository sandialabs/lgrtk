#ifndef J2_HARDENING_HPP_
#define J2_HARDENING_HPP_


namespace lgr {

namespace j2 {

struct Properties {
  double K;
  double G;

  double S0;
  double n;
  double eps0;

  double Svis0;
  double m;
  double eps_dot0;
};


double
HardeningPotential(Properties props, double eqps);

double
FlowStrength(Properties props, double eqps);

double
HardeningRate(Properties props, double eqps);

double
ViscoplasticDualKineticPotential(Properties props, double delta_eqps, double dt);

double
ViscoplasticStress(Properties props, double delta_eqps, double dt);

double
ViscoplasticHardeningRate(Properties props, double delta_eqps, double dt);


} // namespace j2
} // namespace lgr


#endif /* J2_HARDENING_HPP_ */
