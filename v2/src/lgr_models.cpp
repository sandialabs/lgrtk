#include <lgr_models.hpp>
#include <lgr_simulation.hpp>
#include <lgr_linear_elastic.hpp>
#include <lgr_hyper_ep.hpp>
#include <lgr_ideal_gas.hpp>
#include <lgr_mie_gruneisen.hpp>
#include <lgr_neo_hookean.hpp>
#include <lgr_artificial_viscosity.hpp>
#include <lgr_internal_energy.hpp>
#include <lgr_deformation_gradient.hpp>
#include <lgr_scope.hpp>
#include <Omega_h_profile.hpp>

namespace lgr {

Models::Models(Simulation& sim_in)
  :sim(sim_in)
{
}

void Models::setup_material_models_and_modifiers(Omega_h::InputMap& pl) {
  ::lgr::setup(sim.factories.material_model_factories, sim, pl.get_map("material models"), models, "material model");
  for (auto& model_ptr : models) {
    OMEGA_H_CHECK((model_ptr->exec_stages() & AT_MATERIAL_MODEL) != 0);
  }
  if (models.empty()) Omega_h_fail("no material models defined!\n");
  ::lgr::setup(sim.factories.modifier_factories, sim, pl.get_map("modifiers"), models, "modifier");
}

void Models::setup_field_updates() {
  auto const& factories = sim.factories.field_update_factories;
  Omega_h::InputMap dummy_pl;
  for (auto& f_ptr : sim.fields.storage) {
    auto& name = f_ptr->long_name;
    auto it = factories.find(name);
    if (it == factories.end()) continue;
    auto& factory = it->second;
    auto ptr = factory(sim, name, dummy_pl);
    std::unique_ptr<ModelBase> unique_ptr(ptr);
    models.push_back(std::move(unique_ptr));
  }
}

void Models::before_position_update() {
  OMEGA_H_TIME_FUNCTION;
  for (auto& model : models) {
    if ((model->exec_stages() & BEFORE_POSITION_UPDATE) != 0) {
      Scope scope{sim, model->name()};
      model->before_position_update();
    }
  }
}

void Models::at_field_update() {
  OMEGA_H_TIME_FUNCTION;
  for (auto& model : models) {
    if ((model->exec_stages() & AT_FIELD_UPDATE) != 0) {
      Scope scope{sim, model->name()};
      model->at_field_update();
    }
  }
}

void Models::after_field_update() {
  OMEGA_H_TIME_FUNCTION;
  for (auto& model : models) {
    if ((model->exec_stages() & AFTER_FIELD_UPDATE) != 0) {
      Scope scope{sim, model->name()};
      model->after_field_update();
    }
  }
}

void Models::at_material_model() {
  OMEGA_H_TIME_FUNCTION;
  for (auto& model : models) {
    if ((model->exec_stages() & AT_MATERIAL_MODEL) != 0) {
      Scope scope{sim, model->name()};
      model->at_material_model();
    }
  }
}

void Models::after_material_model() {
  OMEGA_H_TIME_FUNCTION;
  for (auto& model : models) {
    if ((model->exec_stages() & AFTER_MATERIAL_MODEL) != 0) {
      Scope scope{sim, model->name()};
      model->after_material_model();
    }
  }
}

void Models::after_correction() {
  OMEGA_H_TIME_FUNCTION;
  for (auto& model : models) {
    if ((model->exec_stages() & AFTER_CORRECTION) != 0) {
      Scope scope{sim, model->name()};
      model->after_correction();
    }
  }
}

template <class Elem>
ModelFactories get_builtin_material_model_factories() {
  ModelFactories out;
  out["linear elastic"] = linear_elastic_factory<Elem>;
  out["hyper elastic-plastic"] = hyper_ep_factory<Elem>;
  out["ideal gas"] = ideal_gas_factory<Elem>;
  out["Mie-Gruneisen"] = mie_gruneisen_factory<Elem>;
  out["neo-Hookean"] = neo_hookean_factory<Elem>;
  return out;
}

template <class Elem>
ModelFactories get_builtin_modifier_factories() {
  ModelFactories out;
  out["artificial viscosity"] = artificial_viscosity_factory<Elem>;
  return out;
}

template <class Elem>
ModelFactories get_builtin_field_update_factories() {
  ModelFactories out;
  out["specific internal energy"] = internal_energy_factory<Elem>;
  out["deformation gradient"] = deformation_gradient_factory<Elem>;
  return out;
}

#define LGR_EXPL_INST(Elem) \
template ModelFactories get_builtin_material_model_factories<Elem>(); \
template ModelFactories get_builtin_modifier_factories<Elem>(); \
template ModelFactories get_builtin_field_update_factories<Elem>();
LGR_EXPL_INST_ELEMS
#undef LGR_EXPL_INST

}
