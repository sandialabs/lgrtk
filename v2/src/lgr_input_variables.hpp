#ifndef LGR_INPUT_VARIABLES_HPP
#define LGR_INPUT_VARIABLES_HPP

#include <Omega_h_input.hpp>
#include <Omega_h_expr.hpp>

namespace lgr {

struct Simulation;

struct InputVariables {
  InputVariables(Simulation& sim_in);
  void setup(Omega_h::InputMap& pl);
  double get_double(Omega_h::InputMap& pl, const char* name, const char* default_expr);
  int get_int(Omega_h::InputMap& pl, const char* name, const char* default_expr);
  Omega_h::ExprEnv env;
};

}

#endif
