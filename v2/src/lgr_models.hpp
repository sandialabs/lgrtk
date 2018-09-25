#ifndef LGR_MODELS_HPP
#define LGR_MODELS_HPP

#include <lgr_factories.hpp>
#include <lgr_model.hpp>
#include <lgr_element_types.hpp>

namespace lgr {

struct Models {
  Simulation& sim;
  std::vector<std::unique_ptr<ModelBase>> models;
  Models(Simulation& sim_in);
  void setup_material_models_and_modifiers(Teuchos::ParameterList& pl);
  void setup_field_updates(); 
  void at_field_update();
  void at_material_model();
  void after_material_model();
  void after_correction();
};

template <class Elem>
ModelFactories get_builtin_material_model_factories();
template <class Elem>
ModelFactories get_builtin_modifier_factories();
template <class Elem>
ModelFactories get_builtin_field_update_factories();

#define LGR_EXPL_INST(Elem) \
extern template ModelFactories get_builtin_material_model_factories<Elem>(); \
extern template ModelFactories get_builtin_modifier_factories<Elem>(); \
extern template ModelFactories get_builtin_field_update_factories<Elem>();
LGR_EXPL_INST_ELEMS
#undef LGR_EXPL_INST

}

#endif
