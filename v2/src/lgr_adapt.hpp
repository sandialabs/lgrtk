#ifndef LGR_ADAPT_HPP
#define LGR_ADAPT_HPP

#include <Omega_h_input.hpp>
#include <lgr_remap.hpp>

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
  double gradation_rate;
  bool should_coarsen_with_expansion;
  bool should_refine_with_eqps;
  Adapter(Simulation& sim);
  void setup(Omega_h::InputMap& pl);
  bool needs_adapt();
  void adapt();
  void coarsen_metric_with_expansion();
  void refine_with_eqps();
  double old_quality;
  double old_length;
};

}  // namespace lgr

#endif
