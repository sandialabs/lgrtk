#include <lgr_mie_gruneisen.hpp>
#include <lgr_scope.hpp>
#include <lgr_simulation.hpp>
#include <lgr_for.hpp>

namespace lgr {

template <class Elem>
struct MieGruneisen : public Model<Elem> {

  FieldIndex rho0_;
  FieldIndex gamma0_;
  FieldIndex cs_;
  FieldIndex s1_;
  FieldIndex specific_internal_energy;

  MieGruneisen(Simulation& sim_in, Teuchos::ParameterList& pl) :
    Model<Elem>(sim_in, pl)
  {
    using std::to_string;

    double rho0, gamma0, c0, s1, e0;
    mie_gruneisen_details::read_and_validate_params(pl, rho0, gamma0, c0, s1, e0);

    this->rho0_ = this->point_define("rho_0", "initial density", 1, to_string(rho0));
    this->gamma0_ = this->point_define("gamma_0", "Gruneisen parameter", 1, to_string(gamma0));
    this->cs_ = this->point_define("c_0", "unshocked sound speed", 1, to_string(c0));
    this->s1_ = this->point_define("S1", "Us/Up ratio", 1, to_string(s1));
    this->specific_internal_energy = this->point_define("e", "specific internal energy", 1, to_string(e0));
  }

  std::uint64_t exec_stages() override final { return AT_MATERIAL_MODEL; }

  char const* name() override final { return "linear elastic"; }

  void update_state() override final {
    auto points_to_grad = this->points_get(this->sim.gradient);
    auto points_to_rho = this->points_get(this->sim.density);

    auto points_to_e = this->points_getset(this->specific_internal_energy);
    auto points_to_rho0 = this->points_get(this->rho0_);
    auto points_to_gamma0 = this->points_get(this->gamma0_);
    auto points_to_cs = this->points_get(this->cs_);
    auto points_to_s1 = this->points_get(this->s1_);

    auto points_to_sigma = this->points_getset(this->sim.stress);
    auto points_to_c = this->points_set(this->sim.wave_speed);
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
      auto e_np1_est = e_nm12 + e_dot_n * (dt_n + (1.0 / 2.0) * dt_np12);

      auto rho0 = points_to_rho0[point];
      auto gamma0 = points_to_gamma0[point];
      auto c0 = points_to_cs[point];
      auto s1 = points_to_s1[point];

      double c;
      double pressure;
      mie_gruneisen_update(rho0, gamma0, c0, s1, rho_n, e_np1_est, pressure, c);

      auto sigma = diagonal(fill_vector<Elem::dim>(-pressure));
      setsymm<Elem>(points_to_sigma, point, sigma);
      points_to_c[point] = c;
      points_to_e[point] = e_np12;
    };
    parallel_for("Mie Gruniesen kernel", this->points(), std::move(functor));
  }
};

template <class Elem>
ModelBase* mie_gruneisen_factory(Simulation& sim, std::string const&, Teuchos::ParameterList& pl) {
  return new MieGruneisen<Elem>(sim, pl);
}

#define LGR_EXPL_INST(Elem) \
template ModelBase* mie_gruneisen_factory<Elem>(Simulation&, std::string const&, Teuchos::ParameterList&);
LGR_EXPL_INST_ELEMS
#undef LGR_EXPL_INST

}
