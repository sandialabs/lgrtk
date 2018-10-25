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
  void flood(double priority_shift = 0.0);
  struct FloodStatus {
    bool some_did_flood;
    bool some_were_bad;
    Omega_h::LOs pull_mapping;
    Omega_h::Reals final_priorities;
  };
  FloodStatus schedule(double priority_shift);
  void flood_once(Omega_h::LOs pull_mapping);
  void schedule_once(
      Omega_h::Bytes elems_can_flood,
      Omega_h::Write<Omega_h::Byte> elems_will_flood,
      Omega_h::Write<Omega_h::LO> pull_mapping,
      Omega_h::Reals old_flooded_priorities,
      Omega_h::Write<double> new_flooded_priorities,
      Omega_h::Reals original_priorities);
};

}

#endif
