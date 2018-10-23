#ifndef LGR_COMPARISON_HPP
#define LGR_COMPARISON_HPP

#include <Omega_h_input.hpp>

namespace lgr {

struct Response;
struct Simulation;

Response* comparison_factory(Simulation& sim, std::string const& name,
    Omega_h::InputMap& pl);

}

#endif
