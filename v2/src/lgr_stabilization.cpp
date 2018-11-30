#include <lgr_stabilization.hpp>
#include <lgr_model.hpp>
#include <lgr_simulation.hpp>
#include <lgr_for.hpp>
#include <Omega_h_print.hpp>

namespace lgr {

template <class Elem>
struct Stabilization : public Model<Elem> {
  FieldIndex velocity_stabilization;
  FieldIndex pressure_stabilization;
  double velocity_constant;
  double pressure_constant;
  Stabilization(Simulation& sim_in, Omega_h::InputMap& pl)
    :Model<Elem>(sim_in, pl)
  {
    velocity_stabilization =
      this->point_define("tau_v", "velocity stabilization", 1,
			 RemapType::NONE, pl, "");
    pressure_stabilization =
      this->point_define("tau_p", "pressure stabilization", 1,
			 RemapType::NONE, pl, "");
    velocity_constant = pl.get<double>("velocity constant", "1.0");
    pressure_constant = pl.get<double>("pressure constant", "1.0");
  }
  std::uint64_t exec_stages() override final { return AT_FIELD_UPDATE; }
  char const* name() override final { return "stabilization"; }
  void at_field_update() override final {
    auto const dt = this->sim.dt;
    auto const cfl = this->sim.cfl;
    auto const c_tau_p = this->pressure_constant;
    auto const c_tau_v = this->velocity_constant;
    auto const points_to_tau_v = this->points_set(velocity_stabilization);
    auto const points_to_tau_p = this->points_set(pressure_stabilization);
    auto functor = OMEGA_H_LAMBDA(int const point) {
      auto const tau_v = c_tau_v * dt / cfl;
      auto const tau_p = c_tau_p * dt / cfl;
      points_to_tau_v[point] = tau_v;
      points_to_tau_p[point] = tau_p;
    };
    parallel_for(this->points(), std::move(functor));
  }
};

template <class Elem>
ModelBase* stabilization_factory(Simulation& sim, std::string const&, Omega_h::InputMap& pl) {
  return new Stabilization<Elem>(sim, pl);
}

#define LGR_EXPL_INST(Elem) \
template ModelBase* stabilization_factory<Elem>(Simulation&, std::string const&, Omega_h::InputMap&);
LGR_EXPL_INST_ELEMS
#undef LGR_EXPL_INST

}
