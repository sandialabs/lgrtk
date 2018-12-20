#ifndef LGR_FLOOD_HPP
#define LGR_FLOOD_HPP

#include <Omega_h_input.hpp>
#include <lgr_field_index.hpp>

namespace lgr {

struct Simulation;

struct Flooder {
  Simulation& sim;
  bool enabled;
  int max_depth;
  FieldIndex flood_priority;
  Flooder(Simulation& sim_in);
  void setup(Omega_h::InputMap& pl);
  void flood();
  Omega_h::LOs choose();
  void flood_by_mapping(Omega_h::LOs pull_mapping);
};

}  // namespace lgr

#endif
