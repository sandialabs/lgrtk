#include <lgr_setup.hpp>
#include <lgr_neo_hookean.hpp>
#include <lgr_ideal_gas.hpp>
#include <lgr_artificial_viscosity.hpp>
#include <lgr_cmdline_hist.hpp>
#include <lgr_comparison.hpp>
#include <lgr_deformation_gradient.hpp>
#include <lgr_hyper_ep.hpp>
#include <lgr_internal_energy.hpp>
#include <lgr_joule_heating.hpp>
#include <lgr_ray_trace.hpp>
#include <lgr_compute_scalar.hpp>

namespace lgr {

void add_builtin_setups(Setups& setups) {
  setups.material_models.push_back(setup_neo_hookean);
  setups.material_models.push_back(setup_ideal_gas);
  setups.material_models.push_back(setup_hyper_ep);
  setups.modifiers.push_back(setup_artifical_viscosity);
  setups.modifiers.push_back(setup_joule_heating);
  setups.modifiers.push_back(setup_ray_trace);
  setups.modifiers.push_back(setup_compute_scalar);
  setups.responses.push_back(setup_cmdline_hist);
  setups.responses.push_back(setup_comparison);
  setups.field_updates.push_back(setup_deformation_gradient);
  setups.field_updates.push_back(setup_internal_energy);
}

}
