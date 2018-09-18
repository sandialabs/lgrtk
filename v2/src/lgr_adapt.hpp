#ifndef LGR_ADAPT_HPP
#define LGR_ADAPT_HPP

#include <lgr_remap.hpp>
#include <Omega_h_teuchos.hpp>

namespace lgr {

struct Simulation;

struct Adapter {
  Simulation& sim;
  Omega_h::AdaptOpts opts;
  std::shared_ptr<RemapBase> remap;
  bool should_adapt;
  double trigger_quality;
  double trigger_length_ratio;
  double minimum_length;
  Adapter(Simulation& sim);
  void setup(Teuchos::ParameterList& pl);
  bool adapt();
};

}

#endif
