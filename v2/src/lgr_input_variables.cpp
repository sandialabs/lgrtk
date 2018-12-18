#include <lgr_input_variables.hpp>

namespace lgr {

InputVariables::InputVariables(Simulation&) {}

void InputVariables::setup(Omega_h::InputMap& pl) {
  env = decltype(env)(1, 1);
  for (auto it = pl.map.begin(), end = pl.map.end(); it != end; ++it) {
    auto& name = it->first;
    double const value = get_double(pl, name.c_str(), "");
    env.register_variable(name, Omega_h::any(value));
  }
}

double InputVariables::get_double(
    Omega_h::InputMap& pl, const char* name, const char* default_expr) {
  auto const expr = pl.get<std::string>(name, default_expr);
  Omega_h::ExprOpsReader reader;
  auto op = reader.read_ops(expr);
  auto const value_any = op->eval(this->env);
  return Omega_h::any_cast<double>(value_any);
}

int InputVariables::get_int(
    Omega_h::InputMap& pl, const char* name, const char* default_expr) {
  return static_cast<int>(get_double(pl, name, default_expr));
}

}  // namespace lgr
