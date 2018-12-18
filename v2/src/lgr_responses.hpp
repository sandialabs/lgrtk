#ifndef LGR_RESPONSES_HPP
#define LGR_RESPONSES_HPP

#include <lgr_factories.hpp>
#include <lgr_response.hpp>

namespace lgr {

struct Simulation;

struct Responses {
  Simulation& sim;
  std::vector<std::unique_ptr<Response>> storage;
  Responses(Simulation& sim_in);
  void setup(Omega_h::InputList& pl);
  void evaluate();
  double next_event(double time);
};

ResponseFactories get_builtin_response_factories();

}  // namespace lgr

#endif
