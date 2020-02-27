#pragma once

namespace lgr {

namespace j2 {

struct Properties {
  double K;
  double G;

  double Y0;
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
