#ifndef LGR_IDEAL_GAS_HPP
#define LGR_IDEAL_GAS_HPP

#include <Omega_h_input.hpp>
#include <lgr_math.hpp>

namespace lgr {

OMEGA_H_INLINE void ideal_gas_update(double const gamma, double const density,
    double const specific_internal_energy, double& pressure,
    double& wave_speed) {
  OMEGA_H_CHECK(density > 0.0);
  OMEGA_H_CHECK(specific_internal_energy > 0.0);
  pressure = (gamma - 1.) * density * specific_internal_energy;
  auto const bulk_modulus = gamma * pressure;
  wave_speed = std::sqrt(bulk_modulus / density);
  OMEGA_H_CHECK(wave_speed > 0.0);
}

struct Simulation;

void setup_ideal_gas(Simulation& sim, Omega_h::InputMap& pl);

}  // namespace lgr

#endif
