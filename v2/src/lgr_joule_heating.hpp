#ifndef LGR_JOULE_HEATING_HPP
#define LGR_JOULE_HEATING_HPP

#include <Omega_h_input.hpp>

namespace lgr {

struct Simulation;

void setup_joule_heating(Simulation& sim, Omega_h::InputMap& pl);

}  // namespace lgr

#endif
