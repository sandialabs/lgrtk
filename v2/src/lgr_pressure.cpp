#include <lgr_pressure.hpp>
#include <lgr_for.hpp>
#include <lgr_model.hpp>
#include <lgr_simulation.hpp>

namespace lgr {

struct Pressure : public ModelBase {
  FieldIndex pressure;
  Pressure(Simulation& sim_in, Omega_h::InputMap& pl)
      : ModelBase(sim_in, pl)
  {
    pressure = this->point_define("p", "pressure", 1, RemapType::NONE, "");
  }
  std::uint64_t exec_stages() override final { return BEFORE_SECONDARIES; }
  char const* name() override final { return "compute pressure"; }
  void before_secondaries() override final {
    auto const points_to_sigma = this->sim.get(this->sim.stress);
    auto const points_to_p = this->sim.set(this->pressure);
    auto functor = OMEGA_H_LAMBDA(int const point) {
      auto const sigma = getstress(points_to_sigma, point);
      auto const p = -trace(sigma);
      points_to_p[point] = p;
    };
    parallel_for(this->points(), std::move(functor));
  }
  void out_of_line_virtual_function() override final;
};

void Pressure::out_of_line_virtual_function() {
}

ModelBase* pressure_factory(
    Simulation& sim, std::string const&, Omega_h::InputMap& pl) {
  return new Pressure(sim, pl);
}

}  // namespace lgr
