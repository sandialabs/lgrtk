#ifndef LGR_RESPONSES_HPP
#define LGR_RESPONSES_HPP

#include <lgr_response.hpp>
#include <lgr_factories.hpp>

namespace lgr {

struct Simulation;

struct Responses {
  Simulation& sim;
  std::vector<std::unique_ptr<Response>> storage;
  Responses(Simulation& sim_in);
  void setup(Teuchos::ParameterList& pl);
  void evaluate();
  double next_event(double time);
};

ResponseFactories get_builtin_response_factories();

}

#endif

