#ifndef LGR_PRESSURE_HPP
#define LGR_PRESSURE_HPP

#include <Omega_h_input.hpp>
#include <string>

namespace lgr {

struct ModelBase;
struct Simulation;

ModelBase* pressure_factory(
    Simulation& sim, std::string const&, Omega_h::InputMap& pl);

}  // namespace lgr

#endif
