#ifndef LGR_EXPRESSION_HPP
#define LGR_EXPRESSION_HPP

#include <Omega_h_expr.hpp>
#include <string>

namespace lgr {

struct Simulation;

struct SingleExpression {
  std::string expr;
  std::string name_tag;
  Omega_h::ExprEnv env;
  Simulation* sim_ptr;
  SingleExpression() : sim_ptr(NULL), expr(""), name_tag("") {}
  SingleExpression(Simulation& sim_in, std::string const& expr_in = "", std::string const& name_in = "")
      : sim_ptr(&sim_in), 
        expr(expr_in), 
        name_tag(name_in) 
  {
    env = decltype(env)(1, 1); // limit to 1d since this is a Single value
    // Register user input variables
    auto& user_env = sim_ptr->input_variables.env;
    for (auto& pair : user_env.variables) {
      auto& name = pair.first;
      auto& value = pair.second;
      env.register_variable(name, value);
    }
  }
  Omega_h::any eval_expr(Omega_h::ExprEnv& aenv, std::string const& expr_in, std::string const& test_name) {
      Omega_h::ExprOpsReader reader;
      auto op = Omega_h::any_cast<Omega_h::OpPtr>(reader.read_string(expr_in, test_name));
      return op->eval(aenv);
  }
  Omega_h::Real evaluate() {
    return Omega_h::any_cast<Omega_h::Real>(eval_expr(env, expr, name_tag));
  }
};

}  // namespace lgr

#endif
