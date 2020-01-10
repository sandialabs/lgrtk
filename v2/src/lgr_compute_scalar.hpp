#ifndef LGR_COMPUTE_SCALAR_HPP
#define LGR_COMPUTE_SCALAR_HPP

#include <Omega_h_input.hpp>
#include <cmath>

namespace lgr {

struct Simulation;

void setup_compute_scalar(Simulation& sim, Omega_h::InputMap& pl);

}  // namespace lgr

#endif
