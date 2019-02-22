#ifndef LGR_NEO_HOOKEAN_HPP
#define LGR_NEO_HOOKEAN_HPP

#include <Omega_h_input.hpp>

namespace lgr {

struct Simulation;

void setup_neo_hookean(Simulation& sim, Omega_h::InputMap& pl);

}  // namespace lgr

#endif
