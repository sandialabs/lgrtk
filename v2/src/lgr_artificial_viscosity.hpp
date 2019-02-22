#ifndef LGR_ARTIFICIAL_VISCOSITY_HPP
#define LGR_ARTIFICIAL_VISCOSITY_HPP

#include <Omega_h_input.hpp>

namespace lgr {

struct Simulation;

void setup_artifical_viscosity(Simulation& sim, Omega_h::InputMap& pl);

}  // namespace lgr

#endif
