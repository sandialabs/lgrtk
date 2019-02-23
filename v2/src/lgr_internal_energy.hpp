#ifndef LGR_INTERNAL_ENERGY_HPP
#define LGR_INTERNAL_ENERGY_HPP

#include <Omega_h_input.hpp>

namespace lgr {

struct Simulation;

void setup_internal_energy(Simulation& sim, Omega_h::InputMap&);

}  // namespace lgr

#endif
