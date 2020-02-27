#include <Omega_h_map.hpp>
#include <lgr_compute_scalar.hpp>
#include <lgr_scalar.hpp>
#include <lgr_simulation.hpp>
#include <Omega_h_simplex.hpp>
#include <Omega_h_shape.hpp>
#include <Omega_h_array_ops.hpp>
#include <lgr_for.hpp>
#include <string>
#include <map>
#include <cmath>
#include <sstream>

namespace lgr {

struct NameExpressionPair {
   std::string name = "";
   std::string expression = "";
   bool integrate = false;
   double value = std::nan("1");
   double last_value = std::nan("1");
   double integral = 0.0;
};

template <class Elem>
struct ComputeScalar : public Model<Elem> {
  using Model<Elem>::sim;
  std::vector<NameExpressionPair> named_expressions;
  ComputeScalar(Simulation& sim_in, Omega_h::InputMap& pl)
      : Model<Elem>(sim_in, pl) {
      // Get user input
      if (pl.is_list("expressions")) {
          auto& pairs_pl = pl.get_list("expressions");
          for (int i=0; i < pairs_pl.size(); ++i) {
              NameExpressionPair nep;
              named_expressions.push_back(nep);
          }
          for (int i=0; i < pairs_pl.size(); ++i) {
              if (pairs_pl.is_map(i)) {
                  std::size_t const k = std::size_t(i);
                  auto& pair_pl = pairs_pl.get_map(i);
                  if (pair_pl.is<std::string>("name")) 
                      named_expressions.at(k).name = pair_pl.get<std::string>("name");
                  if (pair_pl.is<std::string>("expression")) 
                      named_expressions.at(k).expression = pair_pl.get<std::string>("expression");
                  if (pair_pl.is<std::string>("integrate")) {
                      auto yes = pair_pl.get<std::string>("integrate");
                      if (yes == "yes" || yes == "YES" || yes == "true" || yes == "TRUE")
                         named_expressions.at(k).integrate = true;
                  }
              }
          }
      }
  }
  std::uint64_t exec_stages() override final { return AFTER_CORRECTION; }
  char const* name() override final { return "compute scalar"; }
  void after_correction() override final {
    // purge all changing global names if we added already (excluding integral)
    for (auto itr = named_expressions.begin(); itr != named_expressions.end(); ++itr) {
       std::string scalar_name = (*itr).name;
       if (! (*itr).integrate)
          sim.globals.remove(scalar_name);
    }
    for (auto itr = named_expressions.begin(); itr != named_expressions.end(); ++itr) {
       std::string scalar_name = (*itr).name;
       std::string parsed_expression = (*itr).expression;
       // Replace already defined names in expression strings
       // Note: This code requires the full 'name' of an expression to not be
       //       a substring of any other expression 'name' 
       for (auto it = sim.globals.data.begin(); it != sim.globals.data.end(); ++it) {
           auto name = (*it).name;
           auto value = (*it).value;
           std::ostringstream streamObj;
           streamObj.precision(16);
           streamObj << std::fixed << value;
           std::string strObj = streamObj.str();
           findAndReplaceAll(parsed_expression, name, strObj);
       }
       // After replacing global 'names' in expression, evaluate expression
       // against any constant 'input variables' and math operations
       auto value = sim.input_variables.get_double(parsed_expression);
       // For values with the integrate = 'yes' flag, use trapezoidal time integration
       // on the 'expression' arguement
       if ((*itr).integrate) {
          if (std::isnan((*itr).last_value))
              (*itr).last_value = value;
          else
              (*itr).last_value = (*itr).value;
          (*itr).value = value;
          double added = ((*itr).value + (*itr).last_value)*sim.dt/2.0;
          (*itr).integral += added;
          value = (*itr).integral;
       }
       // Register the name with globals
       sim.globals.set(scalar_name,value);
    }
  }
  void findAndReplaceAll(
   std::string & data, std::string toSearch, std::string replaceStr) {
	// Get the first occurrence
	size_t pos = data.find(toSearch);
	// Repeat till end is reached
	while( pos != std::string::npos)
	{
		// Replace this occurrence of Sub String
		data.replace(pos, toSearch.size(), replaceStr);
		// Get the next occurrence from the current position
		pos =data.find(toSearch, pos + replaceStr.size());
	}
  }
};

void setup_compute_scalar(Simulation& sim, Omega_h::InputMap& pl) {
  auto& models_pl = pl.get_list("modifiers");
  for (int i = 0; i < models_pl.size(); ++i) {
    auto& model_pl = models_pl.get_map(i);
    if (model_pl.get<std::string>("type") == "compute scalar") {
#define LGR_EXPL_INST(Elem) \
      if (sim.elem_name == Elem::name()) { \
        sim.models.add(new ComputeScalar<Elem>(sim, model_pl)); \
      }
LGR_EXPL_INST_ELEMS
#undef LGR_EXPL_INST
    }
  }
}

}  // namespace lgr
