#include <lgr_internal_energy.hpp>
#include <lgr_model.hpp>
#include <lgr_simulation.hpp>
#include <lgr_for.hpp>

namespace lgr {

template <class Elem>
struct InternalEnergy : public Model<Elem> {
  FieldIndex specific_internal_energy;
  FieldIndex specific_internal_energy_rate;
  InternalEnergy(Simulation& sim_in)
    :Model<Elem>(sim_in, sim_in.fields[sim_in.fields.find("specific internal energy")].class_names)
    ,specific_internal_energy(sim_in.fields.find("specific internal energy"))
  {
    specific_internal_energy_rate =
      this->point_define("e_dot", "specific internal energy rate", 1,
          RemapType::NONE, "0.0");
  }
  std::uint64_t exec_stages() override final { return BEFORE_MATERIAL_MODEL | BEFORE_SECONDARIES | AFTER_CORRECTION; }
  char const* name() override final { return "internal energy"; }
  void before_material_model() override final {
    compute_internal_energy_predictor();
  }
  void before_secondaries() override final {
    backtrack_to_midpoint_internal_energy();
    zero_internal_energy_rate();
  }
  void after_correction() override final {
    contribute_stress_power();
    correct_internal_energy();
  }
  // based on the previous energy and energy rate, compute a predicted
  // energy using forward Euler. this predicted energy is what material models use
  void compute_internal_energy_predictor() {
    OMEGA_H_TIME_FUNCTION;
    auto const points_to_e = this->points_getset(this->specific_internal_energy);
    auto const points_to_e_dot = this->points_get(this->specific_internal_energy_rate);
    auto const dt = this->sim.dt;
    auto functor = OMEGA_H_LAMBDA(int const point) {
      auto const e_dot_n = points_to_e_dot[point];
      auto const e_n = points_to_e[point];
      auto const e_np1_tilde = e_n + e_dot_n * dt;
      points_to_e[point] = e_np1_tilde;
    };
    parallel_for(this->points(), std::move(functor));
  }
  void backtrack_to_midpoint_internal_energy() {
    OMEGA_H_TIME_FUNCTION;
    auto const points_to_e = this->points_getset(this->specific_internal_energy);
    auto const points_to_e_dot = this->points_get(this->specific_internal_energy_rate);
    auto const dt = this->sim.dt;
    auto functor = OMEGA_H_LAMBDA(int const point) {
      auto const e_dot_n = points_to_e_dot[point];
      auto const e_np1_tilde = points_to_e[point];
      auto const e_np12 = e_np1_tilde - 0.5 * dt * e_dot_n;
      points_to_e[point] = e_np1_tilde;
    };
    parallel_for(this->points(), std::move(functor));
  }
  // zero the rate before other models contribute to it
  void zero_internal_energy_rate() {
    OMEGA_H_TIME_FUNCTION;
    auto const points_to_e_dot = this->points_set(this->specific_internal_energy_rate);
    Omega_h::fill(points_to_e_dot, 0.0);
  }
  void contribute_stress_power() {
    OMEGA_H_TIME_FUNCTION;
    auto const points_to_e_dot = this->points_getset(this->specific_internal_energy_rate);
    auto const points_to_grad = this->points_get(this->sim.gradient);
    auto const points_to_rho = this->points_get(this->sim.density);
    auto const points_to_sigma = this->points_get(this->sim.stress);
    auto const elems_to_nodes = this->get_elems_to_nodes();
    auto const nodes_to_v = this->sim.get(this->sim.velocity);
    auto functor = OMEGA_H_LAMBDA(int const point) {
      auto const elem = point / Elem::points;
      auto const elem_nodes = getnodes<Elem>(elems_to_nodes, elem);
      auto const v_n = getvecs<Elem>(nodes_to_v, elem_nodes);
      auto const dN_dxn = getgrads<Elem>(points_to_grad, point);
      auto const dvn_dxn = grad<Elem>(dN_dxn, v_n);
      auto const sigma_n = getsymm<Elem>(points_to_sigma, point);
      auto const e_rho_dot_n = inner_product(dvn_dxn, sigma_n);
      auto const rho_n = points_to_rho[point];
      auto const e_dot_n = e_rho_dot_n / rho_n;
      points_to_e_dot[point] += e_dot_n;
    };
    parallel_for(this->points(), std::move(functor));
  }
  // using the previous midpoint energy and the current energy rate,
  // compute the current energy
  void correct_internal_energy() override final {
    OMEGA_H_TIME_FUNCTION;
    auto const points_to_e = this->points_getset(this->specific_internal_energy);
    auto const points_to_e_dot = this->points_get(this->specific_internal_energy_rate);
    auto const dt = this->sim.dt;
    auto functor = OMEGA_H_LAMBDA(int const point) {
      auto const e_dot_np1 = points_to_e_dot[point];
      auto const e_np12 = points_to_e[point];
      auto const e_np1 = e_np12 + e_dot_np1 * (0.5 * dt);
      points_to_e[point] = e_np1;
    };
    parallel_for(this->points(), std::move(functor));
  }
};

template <class Elem>
ModelBase* internal_energy_factory(Simulation& sim, std::string const&, Omega_h::InputMap&) {
  return new InternalEnergy<Elem>(sim);
}

#define LGR_EXPL_INST(Elem) \
template ModelBase* internal_energy_factory<Elem>(Simulation&, std::string const&, Omega_h::InputMap&);
LGR_EXPL_INST_ELEMS
#undef LGR_EXPL_INST

}
