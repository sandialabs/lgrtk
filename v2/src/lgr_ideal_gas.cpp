#include <lgr_for.hpp>
#include <lgr_ideal_gas.hpp>
#include <lgr_simulation.hpp>

namespace lgr {

template <class Elem>
struct IdealGas : public Model<Elem> {
  FieldIndex heat_capacity_ratio;
  FieldIndex specific_internal_energy;
  IdealGas(Simulation& sim_in, Omega_h::InputMap& pl)
      : Model<Elem>(sim_in, pl) {
    this->specific_internal_energy = this->point_define(
        "e", "specific internal energy", 1, RemapType::PER_UNIT_MASS, pl, "");
    this->heat_capacity_ratio = this->point_define(
        "gamma", "heat capacity ratio", 1, RemapType::PER_UNIT_VOLUME, pl, "");
  }
  std::uint64_t exec_stages() override final { return AT_MATERIAL_MODEL; }
  char const* name() override final { return "ideal gas"; }
  void at_material_model() override final {
    auto const points_to_rho = this->points_get(this->sim.density);
    auto const points_to_e = this->points_get(this->specific_internal_energy);
    auto const points_to_gamma = this->points_get(this->heat_capacity_ratio);
    auto const points_to_sigma = this->points_set(this->sim.stress);
    auto const points_to_c = this->points_set(this->sim.wave_speed);
    auto functor = OMEGA_H_LAMBDA(int point) {
      auto const rho_np1 = points_to_rho[point];
      auto const e_np1_est = points_to_e[point];
      auto const gamma = points_to_gamma[point];
      double c;
      double pressure;
      ideal_gas_update(gamma, rho_np1, e_np1_est, pressure, c);
      auto const sigma = diagonal(fill_vector<3>(-pressure));
      setstress(points_to_sigma, point, sigma);
      points_to_c[point] = c;
    };
    parallel_for(this->points(), std::move(functor));
  }
};

template <class Elem>
ModelBase* ideal_gas_factory(
    Simulation& sim, std::string const&, Omega_h::InputMap& pl) {
  return new IdealGas<Elem>(sim, pl);
}

#define LGR_EXPL_INST(Elem)                                                    \
  template ModelBase* ideal_gas_factory<Elem>(                                 \
      Simulation&, std::string const&, Omega_h::InputMap&);
LGR_EXPL_INST_ELEMS
#undef LGR_EXPL_INST

void setup_ideal_gas(Simulation& sim, Omega_h::InputMap& pl) {
  auto& models_pl = pl.get_list("material models");
  for (int i = 0; i < models_pl.size(); ++i) {
    auto& model_pl = models_pl.get_map(i);
    if (model_pl.get<std::string>("type") == "ideal gas") {
#define LGR_EXPL_INST(Elem) \
      if (sim.elem_name == Elem::name()) { \
        sim.models.add(new IdealGas<Elem>(sim, model_pl)); \
      }
LGR_EXPL_INST_ELEMS
#undef LGR_EXPL_INST
    }
  }
}

}  // namespace lgr
