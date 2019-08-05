#include <lgr_for.hpp>
#include <lgr_void.hpp>
#include <lgr_simulation.hpp>

namespace lgr {

template <class Elem>
struct VoidMaterial : public Model<Elem> {
  VoidMaterial(Simulation& sim_in, Omega_h::InputMap& pl)
      : Model<Elem>(sim_in, pl) {

  }
  std::uint64_t exec_stages() override final { return AT_MATERIAL_MODEL; }
  char const* name() override final { return "void material"; }
  void at_material_model() override final {
     zero_stress_wavespeed();
  }
  // zero the rate before other models contribute to it
  void zero_stress_wavespeed() {
    auto const points_to_sigma = this->points_set(this->sim.stress);
    auto const points_to_c = this->points_set(this->sim.wave_speed);
    auto functor = OMEGA_H_LAMBDA(int point) {
        points_to_sigma[point] = 0.0;
        points_to_c[point] = 0.0;
    };
    parallel_for(this->points(), std::move(functor));
  }
};

void setup_void_material(Simulation& sim, Omega_h::InputMap& pl) {
  auto& models_pl = pl.get_list("material models");
  for (int i = 0; i < models_pl.size(); ++i) {
    auto& model_pl = models_pl.get_map(i);
    if (model_pl.get<std::string>("type") == "void") {
#define LGR_EXPL_INST(Elem) \
      if (sim.elem_name == Elem::name()) { \
        sim.models.add(new VoidMaterial<Elem>(sim, model_pl)); \
      }
LGR_EXPL_INST_ELEMS
#undef LGR_EXPL_INST
    }
  }
}

}  // namespace lgr
