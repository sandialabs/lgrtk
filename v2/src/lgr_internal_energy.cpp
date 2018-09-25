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
          RemapType::PER_UNIT_VOLUME, "");
    this->sim.fields[this->sim.stress].default_value = "symm(0.0)";
    this->sim.fields[this->sim.acceleration].default_value = "vector(0.0)";
  }
  std::uint64_t exec_stages() override final { return AT_FIELD_UPDATE; }
  char const* name() override final { return "internal energy"; }
  void at_field_update() override final {
    auto points_to_grad = this->points_get(this->sim.gradient);
    auto points_to_rho = this->points_get(this->sim.density);
    auto points_to_sigma = this->points_get(this->sim.stress);
    auto points_to_e = this->points_getset(this->specific_internal_energy);
    auto points_to_e_dot = this->points_set(this->specific_internal_energy_rate);
    auto elems_to_nodes = this->get_elems_to_nodes();
    auto nodes_to_v = this->sim.get(this->sim.velocity);
    auto nodes_to_a = this->sim.get(this->sim.acceleration);
    auto dt_nm12 = this->sim.prev_dt;
    auto dt_np12 = this->sim.dt;
    auto dt_n = (1.0 / 2.0) * (dt_np12 + dt_nm12);
    auto functor = OMEGA_H_LAMBDA(int point) {
      auto elem = point / Elem::points;
      auto elem_nodes = getnodes<Elem>(elems_to_nodes, elem);
      auto v_np12 = getvecs<Elem>(nodes_to_v, elem_nodes);
      auto a_n = getvecs<Elem>(nodes_to_a, elem_nodes);
      auto v_nm12 = v_np12 - dt_n * a_n;
      auto vavg = (1.0 / 2.0) * (v_np12 + v_nm12); // not the same as v_n if dt != prev_dt
      auto dN_dxnp1 = getgrads<Elem>(points_to_grad, point);
      auto dvavg_dxnp1 = grad<Elem>(dN_dxnp1, vavg);
      auto dvnp12_dxnp1 = grad<Elem>(dN_dxnp1, v_np12);
      auto I = identity_matrix<Elem::dim, Elem::dim>();
      auto dxn_dxnp1 = I - dt_np12 * dvnp12_dxnp1;
      auto dxnp1_dxn = invert(dxn_dxnp1);
      auto dvavg_dxn = dxnp1_dxn * dvavg_dxnp1;
      auto sigma_n = getsymm<Elem>(points_to_sigma, point);
      auto e_rho_dot_n = inner_product(dvavg_dxn, sigma_n);
      auto rho_np1 = points_to_rho[point];
      auto rho_n = determinant(dxnp1_dxn) * rho_np1;
      auto e_dot_n = e_rho_dot_n / rho_n;
      auto e_nm12 = points_to_e[point];
      auto e_np12 = e_nm12 + e_dot_n * dt_n;
      points_to_e[point] = e_np12;
      points_to_e_dot[point] = e_dot_n;
    };
    parallel_for("internal energy kernel", this->points(), std::move(functor));
  }
};

template <class Elem>
ModelBase* internal_energy_factory(Simulation& sim, std::string const&, Teuchos::ParameterList&) {
  return new InternalEnergy<Elem>(sim);
}

#define LGR_EXPL_INST(Elem) \
template ModelBase* internal_energy_factory<Elem>(Simulation&, std::string const&, Teuchos::ParameterList&);
LGR_EXPL_INST_ELEMS
#undef LGR_EXPL_INST

}
