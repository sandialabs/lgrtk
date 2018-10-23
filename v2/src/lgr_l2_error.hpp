#ifndef LGR_L2_ERROR_HPP
#define LGR_L2_ERROR_HPP

#include <Omega_h_input.hpp>

namespace lgr {

struct Scalar;
struct Simulation;

Scalar* l2_error_factory(Simulation& sim, std::string const& name, Omega_h::InputMap& pl);

}

#endif
