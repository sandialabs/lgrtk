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
  struct FloodStatus {
    bool some_did_flood;
    bool some_were_bad;
  };
  FloodStatus flood_once(int depth, double up_to_priority);
};

}

#endif
