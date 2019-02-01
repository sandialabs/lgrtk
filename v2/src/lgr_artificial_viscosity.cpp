#include <lgr_artificial_viscosity.hpp>
#include <lgr_for.hpp>
#include <lgr_simulation.hpp>

namespace lgr {

OMEGA_H_INLINE void artificial_viscosity_update(double const linear,
    double const quadratic, double const h_min, double const h_max,
    double const density, Matrix<3, 3> const grad_v, Matrix<3, 3>& stress,
    double& wave_speed) {
  auto const V_dot = trace(grad_v);
  auto const kinematic =
      quadratic * std::abs(V_dot) * square(h_max) + linear * wave_speed * h_max;
  auto const symm_grad_v = (1. / 2.) * (grad_v + transpose(grad_v));
  stress += density * kinematic * symm_grad_v;
  auto const squiggle = kinematic / (wave_speed * h_min);
  wave_speed *= (std::sqrt(1.0 + square(squiggle)) + squiggle);
}

template <class Elem>
struct ArtificialViscosity : public Model<Elem> {
  FieldIndex linear;
  FieldIndex quadratic;
  ArtificialViscosity(Simulation& sim_in, Omega_h::InputMap& pl)
      : Model<Elem>(sim_in, pl) {
    this->linear = this->point_define("nu_l", "linear artificial viscosity", 1,
        RemapType::PER_UNIT_VOLUME, pl, "");
    this->quadratic =
        this->point_define("nu_q", "quadratic artificial viscosity", 1,
            RemapType::PER_UNIT_VOLUME, pl, "");
  }
  std::uint64_t exec_stages() override final { return AFTER_MATERIAL_MODEL; }
  char const* name() override final { return "artificial viscosity"; }
  void after_material_model() override final {
    auto const points_to_nu_l = this->points_get(this->linear);
    auto const points_to_nu_q = this->points_get(this->quadratic);
    auto const points_to_grad = this->points_get(this->sim.gradient);
    auto const points_to_sigma = this->points_getset(this->sim.stress);
    auto const points_to_c = this->points_getset(this->sim.wave_speed);
    auto const points_to_rho = this->points_get(this->sim.density);
    auto const elems_to_nodes = this->get_elems_to_nodes();
    auto const nodes_to_v = this->sim.get(this->sim.velocity);
    auto const elems_to_h_min = this->sim.get(this->sim.time_step_length);
    auto const elems_to_h_max = this->sim.get(this->sim.viscosity_length);
    auto functor = OMEGA_H_LAMBDA(int point) {
      auto const elem = point / Elem::points;
      auto const elem_nodes = getnodes<Elem>(elems_to_nodes, elem);
      auto const v = getvecs<Elem>(nodes_to_v, elem_nodes);
      auto const dN_dxnp1 = getgrads<Elem>(points_to_grad, point);
      auto const grad_v = resize<3>(grad<Elem>(dN_dxnp1, v));
      auto const nu_l = points_to_nu_l[point];
      auto const nu_q = points_to_nu_q[point];
      auto const h_min = elems_to_h_min[elem];
      auto const h_max = elems_to_h_max[elem];
      auto const rho = points_to_rho[point];
      auto sigma = getstress(points_to_sigma, point);
      auto c = points_to_c[point];
      artificial_viscosity_update(
          nu_l, nu_q, h_min, h_max, rho, grad_v, sigma, c);
      setstress(points_to_sigma, point, sigma);
      points_to_c[point] = c;
    };
    parallel_for(this->points(), std::move(functor));
  }
};

template <class Elem>
ModelBase* artificial_viscosity_factory(
    Simulation& sim, std::string const&, Omega_h::InputMap& pl) {
  return new ArtificialViscosity<Elem>(sim, pl);
}

#define LGR_EXPL_INST(Elem)                                                    \
  template ModelBase* artificial_viscosity_factory<Elem>(                      \
      Simulation&, std::string const&, Omega_h::InputMap&);
LGR_EXPL_INST_ELEMS
#undef LGR_EXPL_INST

}  // namespace lgr
