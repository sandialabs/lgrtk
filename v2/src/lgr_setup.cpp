#include <lgr_setup.hpp>
#include <lgr_neo_hookean.hpp>
#include <lgr_ideal_gas.hpp>
#include <lgr_artificial_viscosity.hpp>
#include <lgr_cmdline_hist.hpp>
#include <lgr_comparison.hpp>

namespace lgr {

void add_builtin_setups(Setups& setups) {
  setups.material_models.push_back(setup_neo_hookean);
  setups.material_models.push_back(setup_ideal_gas);
  setups.modifiers.push_back(setup_artifical_viscosity);
  setups.responses.push_back(setup_cmdline_hist);
  setups.responses.push_back(setup_comparison);
}

}
