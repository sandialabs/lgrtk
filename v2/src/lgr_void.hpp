#ifndef LGR_VOID_HPP
#define LGR_VOID_HPP

#include <Omega_h_input.hpp>

namespace lgr {

struct Simulation;

void setup_void_material(Simulation& sim, Omega_h::InputMap& pl);

}  // namespace lgr

#endif
