#ifndef LGR_RESPONSE_HPP
#define LGR_RESPONSE_HPP

#include <lgr_when.hpp>
#include <memory>

namespace lgr {

struct Simulation;

struct Response {
  Simulation& sim;
  std::unique_ptr<When> when;
  Response(Simulation& sim_in, Omega_h::InputMap& pl);
  virtual ~Response() = default;
  virtual void out_of_line_virtual_method();
  virtual void respond() = 0;
};

}

#endif
