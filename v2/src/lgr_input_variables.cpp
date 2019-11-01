#include <lgr_input_variables.hpp>
#include <Omega_h_fail.hpp>
#ifndef LGR_DISABLE_APREPRO
#include <aprepro.h>
#endif
#include <fstream>
#include <vector>

namespace lgr {

InputVariables::InputVariables(Simulation&) {}

void InputVariables::setup(Omega_h::InputMap& pl) {
  env = decltype(env)(1, 1);
  for (auto& name : pl) {
    if (name == "aprepro filename") {
       auto filename = get_string(pl, name.c_str(), "");
       register_aprepro_vars(filename);
    } else {
       double const value = get_double(pl, name.c_str(), "");
       env.register_variable(name, Omega_h::any(value));
    }
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

std::string InputVariables::get_string(
    Omega_h::InputMap& pl, const char* name, const char* default_expr) {
  return pl.get<std::string>(name, default_expr);
}

void InputVariables::register_aprepro_vars(std::string& filename) {
#ifndef LGR_DISABLE_APREPRO
   SEAMS::Aprepro aprepro;

   std::fstream infile(filename.c_str());
   if (!infile.good())
     Omega_h_fail("APREPRO: Could not open file: %s", filename.c_str());

   bool result = aprepro.parse_stream(infile, filename);
   if (result) {
       std::vector<std::string> names = aprepro.get_variable_names();
       for (auto it = names.begin(); it != names.end(); ++it) {
           auto name = *it;
           auto var = aprepro.getsym(name);
           // Unless we link SEACAS with BISON, we have to assume we know string variables
           if (name == "Material") {
              auto value = var->value.svar;
              printf("Found APREPRO name %s with value %s \n", name.c_str(), value.c_str());
              env.register_variable(name, Omega_h::any(value));
           // Everything else is just a double then
           } else {
              auto value = var->value.var;
              printf("Found APREPRO name %s with value %f \n", name.c_str(), value);
              env.register_variable(name, Omega_h::any(value));
           }
       }
   }
#endif
}

}  // namespace lgr
