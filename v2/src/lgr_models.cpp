#include <Omega_h_profile.hpp>
#include <lgr_artificial_viscosity.hpp>
#include <lgr_pressure.hpp>
#include <lgr_deformation_gradient.hpp>
#include <lgr_hyper_ep.hpp>
#include <lgr_ideal_gas.hpp>
#include <lgr_internal_energy.hpp>
#include <lgr_joule_heating.hpp>
#include <lgr_linear_elastic.hpp>
#include <lgr_mie_gruneisen.hpp>
#include <lgr_models.hpp>
#include <lgr_neo_hookean.hpp>
#include <lgr_nodal_pressure.hpp>
#include <lgr_scope.hpp>
#include <lgr_simulation.hpp>
#include <lgr_stvenant_kirchhoff.hpp>
#include <lgr_anti_lock.hpp>
#include <lgr_indset.hpp>

namespace lgr {

Models::Models(Simulation& sim_in) : sim(sim_in) {}

void Models::setup_material_models_and_modifiers(Omega_h::InputMap& pl) {
  ::lgr::setup(sim.factories.material_model_factories, sim,
      pl.get_list("material models"), models, "material model");
  for (auto& model_ptr : models) {
    OMEGA_H_CHECK((model_ptr->exec_stages() & AT_MATERIAL_MODEL) != 0);
  }
//if (models.empty()) Omega_h_fail("no material models defined!\n");
  ::lgr::setup(sim.factories.modifier_factories, sim, pl.get_list("modifiers"),
      models, "modifier");
}

void Models::setup_field_updates() {
  auto const& factories = sim.factories.field_update_factories;
  Omega_h::InputMap dummy_pl;
  // this can't be a range-based for loop because some field update
  // models create fields which alters the sim.fields.storage vector
  // which invalidates most iterators to it including the one used by
  // the range based for loop
  for (std::size_t i = 0; i < sim.fields.storage.size(); ++i) {
    auto& f_ptr = sim.fields.storage[i];
    auto& name = f_ptr->long_name;
    auto it = factories.find(name);
    if (it == factories.end()) continue;
    auto& factory = it->second;
    auto ptr = factory(sim, name, dummy_pl);
    std::unique_ptr<ModelBase> unique_ptr(ptr);
    models.push_back(std::move(unique_ptr));
  }
}

void Models::learn_disc() {
  OMEGA_H_TIME_FUNCTION;
  for (auto& model : models) {
    model->learn_disc();
  }
}

#define LGR_STAGE_DEF(lowercase, uppercase)                                    \
  void Models::lowercase() {                                                   \
    OMEGA_H_TIME_FUNCTION;                                                     \
    for (auto& model : models) {                                               \
      if ((model->exec_stages() & uppercase) != 0) {                           \
        Scope scope{sim, model->name()};                                       \
        model->lowercase();                                                    \
      }                                                                        \
    }                                                                          \
  }
LGR_STAGE_DEF(after_configuration, AFTER_CONFIGURATION)
LGR_STAGE_DEF(before_field_update, BEFORE_FIELD_UPDATE)
LGR_STAGE_DEF(at_field_update, AT_FIELD_UPDATE)
LGR_STAGE_DEF(after_field_update, AFTER_FIELD_UPDATE)
LGR_STAGE_DEF(before_material_model, BEFORE_MATERIAL_MODEL)
LGR_STAGE_DEF(at_material_model, AT_MATERIAL_MODEL)
LGR_STAGE_DEF(after_material_model, AFTER_MATERIAL_MODEL)
LGR_STAGE_DEF(before_secondaries, BEFORE_SECONDARIES)
LGR_STAGE_DEF(at_secondaries, AT_SECONDARIES)
LGR_STAGE_DEF(after_secondaries, AFTER_SECONDARIES)
LGR_STAGE_DEF(after_correction, AFTER_CORRECTION)
#undef LGR_STAGE_DEF

template <class Elem>
ModelFactories get_builtin_material_model_factories() {
  ModelFactories out;
  out["linear elastic"] = linear_elastic_factory<Elem>;
  out["hyper elastic-plastic"] = hyper_ep_factory<Elem>;
  out["ideal gas"] = ideal_gas_factory<Elem>;
  out["Mie-Gruneisen"] = mie_gruneisen_factory<Elem>;
  out["neo-Hookean"] = neo_hookean_factory<Elem>;
  out["StVenant-Kirchhoff"] = stvenant_kirchhoff_factory<Elem>;
  return out;
}

template <class Elem>
ModelFactories get_builtin_modifier_factories() {
  ModelFactories out;
  out["artificial viscosity"] = artificial_viscosity_factory<Elem>;
  out["Joule heating"] = joule_heating_factory<Elem>;
  out["nodal pressure"] = nodal_pressure_factory<Elem>;
  out["compute pressure"] = pressure_factory;
  out["average J over points"] = average_J_over_points_factory<Elem>;
  out["average pressure over points"] = average_pressure_over_points_factory<Elem>;
  out["average internal energy over points"] = average_internal_energy_over_points_factory<Elem>;
  out["average density over points"] = average_density_over_points_factory<Elem>;
  out["average J over independent set"] = average_J_over_independent_set_factory<Elem>;
  out["average pressure over independent set"] = average_pressure_over_independent_set_factory<Elem>;
  out["independent set"] = independent_set_factory<Elem>;
  return out;
}

template <class Elem>
ModelFactories get_builtin_field_update_factories() {
  ModelFactories out;
  out["specific internal energy"] = internal_energy_factory<Elem>;
  out["deformation gradient"] = deformation_gradient_factory<Elem>;
  return out;
}

#define LGR_EXPL_INST(Elem)                                                    \
  template ModelFactories get_builtin_material_model_factories<Elem>();        \
  template ModelFactories get_builtin_modifier_factories<Elem>();              \
  template ModelFactories get_builtin_field_update_factories<Elem>();
LGR_EXPL_INST_ELEMS
#undef LGR_EXPL_INST

}  // namespace lgr
