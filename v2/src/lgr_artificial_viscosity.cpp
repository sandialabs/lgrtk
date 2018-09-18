#include <lgr_artificial_viscosity.hpp>
#include <lgr_scope.hpp>
#include <lgr_simulation.hpp>
#include <lgr_for.hpp>

namespace lgr {

template <int dim>
OMEGA_H_INLINE void artificial_viscosity_update(
    double linear,
    double quadratic,
    double h_min,
    double h_max,
    double density,
    Matrix<dim, dim> velocity_gradient,
    Matrix<dim, dim>& stress,
    double& wave_speed) {
  auto volume_rate = trace(velocity_gradient);
  auto kinematic =
    quadratic * std::abs(volume_rate) * square(h_max) +
    linear * wave_speed * h_max;
  auto symm_vel_grad = (1./2.) * (
      velocity_gradient + transpose(velocity_gradient));
  stress += density * kinematic * symm_vel_grad;
  auto squiggle = kinematic / (wave_speed * h_min);
  wave_speed *= (std::sqrt(1.0 + square(squiggle)) + squiggle);
}

template <class Elem>
struct ArtificialViscosity : public Model<Elem> {
  FieldIndex linear;
  FieldIndex quadratic;
  ArtificialViscosity(Simulation& sim_in, Teuchos::ParameterList& pl):Model<Elem>(sim_in, pl) {
    this->linear = this->point_define("nu_l", "linear artificial viscosity", 1, RemapType::PER_UNIT_VOLUME, "");
    this->quadratic = this->point_define("nu_q", "quadratic artificial viscosity", 1, RemapType::PER_UNIT_VOLUME, "");
  }
  ModelOrder order() override final { return AFTER_MATERIAL_MODEL; }
  char const* name() override final { return "artificial viscosity"; }
  void update_state() override final {
    auto points_to_nu_l = this->points_get(this->linear);
    auto points_to_nu_q = this->points_get(this->quadratic);
    auto points_to_grad = this->points_get(this->sim.gradient);
    auto points_to_sigma = this->points_getset(this->sim.stress);
    auto points_to_c = this->points_getset(this->sim.wave_speed);
    auto points_to_rho = this->points_get(this->sim.density);
    auto elems_to_nodes = this->get_elems_to_nodes();
    auto nodes_to_v = this->sim.get(this->sim.velocity);
    auto elems_to_h_min = this->sim.get(this->sim.time_step_length);
    auto elems_to_h_max = this->sim.get(this->sim.viscosity_length);
    auto functor = OMEGA_H_LAMBDA(int point) {
      auto elem = point / Elem::points;
      auto elem_nodes = getnodes<Elem>(elems_to_nodes, elem);
      auto v = getvecs<Elem>(nodes_to_v, elem_nodes);
      auto dN_dxnp1 = getgrads<Elem>(points_to_grad, point);
      auto grad_v = grad<Elem>(dN_dxnp1, v);
      auto nu_l = points_to_nu_l[point];
      auto nu_q = points_to_nu_q[point];
      auto h_min = elems_to_h_min[elem];
      auto h_max = elems_to_h_max[elem];
      auto rho = points_to_rho[point];
      auto sigma = getsymm<Elem>(points_to_sigma, point);
      auto c = points_to_c[point];
      artificial_viscosity_update(nu_l, nu_q, h_min, h_max,
          rho, grad_v, sigma, c);
      setsymm<Elem>(points_to_sigma, point, sigma);
      points_to_c[point] = c;
    };
    parallel_for("artificial viscosity kernel",
        this->points(), std::move(functor));
  }
};

template <class Elem>
ModelBase* artificial_viscosity_factory(
    Simulation& sim, std::string const&,
    Teuchos::ParameterList& pl) {
  return new ArtificialViscosity<Elem>(sim, pl);
}

#define LGR_EXPL_INST(Elem) \
template ModelBase* \
artificial_viscosity_factory<Elem>( \
    Simulation&, std::string const&, Teuchos::ParameterList&);
LGR_EXPL_INST_ELEMS
#undef LGR_EXPL_INST

}
