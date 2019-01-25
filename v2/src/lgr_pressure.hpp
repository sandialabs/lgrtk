#ifndef LGR_PRESSURE_HPP
#define LGR_PRESSURE_HPP

#include <string>
#include <Omega_h_input.hpp>

namespace lgr {

struct ModelBase;
struct Simulation;

ModelBase* pressure_factory(
    Simulation& sim, std::string const&, Omega_h::InputMap& pl);

}

#endif
