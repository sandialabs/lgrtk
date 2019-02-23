#ifndef LGR_DEFORMATION_GRADIENT_HPP
#define LGR_DEFORMATION_GRADIENT_HPP

#include <Omega_h_input.hpp>

namespace lgr {

struct Simulation;

void setup_deformation_gradient(Simulation& sim, Omega_h::InputMap& pl);

}  // namespace lgr

#endif
