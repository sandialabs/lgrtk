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
          RemapType::PER_UNIT_VOLUME, "0.0");
  }
  std::uint64_t exec_stages() override final { return BEFORE_POSITION_UPDATE | AFTER_CORRECTION; }
  char const* name() override final { return "internal energy"; }
  void before_position_update() override final {
    auto const points_to_grad = this->points_get(this->sim.gradient);
    auto const points_to_rho = this->points_get(this->sim.density);
    auto const points_to_sigma = this->points_get(this->sim.stress);
    auto const points_to_e = this->points_getset(this->specific_internal_energy);
    auto const points_to_e_dot = this->points_set(this->specific_internal_energy_rate);
    auto const elems_to_nodes = this->get_elems_to_nodes();
    auto const nodes_to_v = this->sim.get(this->sim.velocity);
    auto const dt = this->sim.dt;
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
      auto const e_n = points_to_e[point];
      auto const e_np12 = e_n + e_dot_n * (0.5 * dt);
      points_to_e[point] = e_np12;
      points_to_e_dot[point] = e_dot_n;
    };
    parallel_for(this->points(), std::move(functor));
  }
  void after_correction() override final {
    auto const points_to_grad = this->points_get(this->sim.gradient);
    auto const points_to_rho = this->points_get(this->sim.density);
    auto const points_to_sigma = this->points_get(this->sim.stress);
    auto const points_to_e = this->points_getset(this->specific_internal_energy);
    auto const points_to_e_dot = this->points_set(this->specific_internal_energy_rate);
    auto const elems_to_nodes = this->get_elems_to_nodes();
    auto const nodes_to_v = this->sim.get(this->sim.velocity);
    auto const dt = this->sim.dt;
    auto functor = OMEGA_H_LAMBDA(int const point) {
      auto const elem = point / Elem::points;
      auto const elem_nodes = getnodes<Elem>(elems_to_nodes, elem);
      auto const v_np1 = getvecs<Elem>(nodes_to_v, elem_nodes);
      auto const dN_dxnp1 = getgrads<Elem>(points_to_grad, point);
      auto const dvnp1_dxnp1 = grad<Elem>(dN_dxnp1, v_np1);
      auto const sigma_np1 = getsymm<Elem>(points_to_sigma, point);
      auto const e_rho_dot_np1 = inner_product(dvnp1_dxnp1, sigma_np1);
      auto const rho_np1 = points_to_rho[point];
      auto const e_dot_np1 = e_rho_dot_np1 / rho_np1;
      auto const e_np12 = points_to_e[point];
      auto const e_np1 = e_np12 + e_dot_np1 * (0.5 * dt);
      points_to_e[point] = e_np1;
      points_to_e_dot[point] = e_dot_np1;
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
