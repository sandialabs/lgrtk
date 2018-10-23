#ifndef LGR_OSH_OUTPUT_HPP
#define LGR_OSH_OUTPUT_HPP

#include <Omega_h_input.hpp>

namespace lgr {

struct Response;
struct Simulation;

Response* osh_output_factory(Simulation& sim, std::string const&, Omega_h::InputMap& pl);

}

#endif
