#ifndef LGR_CONDITION_HPP
#define LGR_CONDITION_HPP

#include <lgr_class_names.hpp>
#include <Omega_h_expr.hpp>
#include <lgr_when.hpp>

namespace lgr {

struct Field;
struct Fields;
struct Support;
struct Subsets;
struct SubsetBridge;
struct Disc;
struct Simulation;

struct Condition {
  Field* field;
  std::string str;
  Support* support;
  Omega_h::ExprEnv env;
  std::shared_ptr<Omega_h::ExprOp> op;
  std::unique_ptr<When> when;
  bool needs_reeval;
  bool needs_coords;
  bool uses_old_vals;
  SubsetBridge* bridge;
  Omega_h::Read<double> cached_values;
  Simulation* sim_ptr;
  void init();
  Condition(Field*, Simulation&, std::string const& str_in, Support*, When*);
  Condition(Field* field_in, Simulation&, Omega_h::InputMap& pl);
  void forget_disc();
  void learn_disc();
  double next_event(double time);
  void apply(double prev_time, double time,
      Omega_h::Read<double> node_coords, Fields& fields);
  void apply(double time, Omega_h::Read<double> node_coords, Fields& fields);
};

}

#endif
