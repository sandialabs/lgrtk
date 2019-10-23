#ifndef LGR_INPUT_VARIABLES_HPP
#define LGR_INPUT_VARIABLES_HPP

#include <Omega_h_expr.hpp>
#include <Omega_h_input.hpp>
#include <string>

namespace lgr {

struct Simulation;

struct InputVariables {
  InputVariables(Simulation& sim_in);
  void setup(Omega_h::InputMap& pl);
  double get_double(
      Omega_h::InputMap& pl, const char* name, const char* default_expr);
  int get_int(
      Omega_h::InputMap& pl, const char* name, const char* default_expr);
  std::string get_string(
      Omega_h::InputMap& pl, const char* name, const char* default_expr);
  void register_aprepro_vars(std::string& filename);
  Omega_h::ExprEnv env;
};

}  // namespace lgr

#endif
