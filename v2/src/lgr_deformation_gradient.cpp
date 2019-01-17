#include <Omega_h_print.hpp>
#include <lgr_deformation_gradient.hpp>
#include <lgr_for.hpp>
#include <lgr_model.hpp>
#include <lgr_simulation.hpp>

namespace lgr {

template <class Elem>
struct DeformationGradient : public Model<Elem> {
  FieldIndex deformation_gradient;
  DeformationGradient(Simulation& sim_in)
      : Model<Elem>(
            sim_in, sim_in.fields[sim_in.fields.find("deformation gradient")]
                        .class_names),
        deformation_gradient(sim_in.fields.find("deformation gradient")) {}
  std::uint64_t exec_stages() override final { return AT_FIELD_UPDATE; }
  char const* name() override final { return "deformation gradient"; }
  void at_field_update() override final {
    auto const points_to_grad = this->points_get(this->sim.gradient);
    auto const points_to_F = this->points_getset(this->deformation_gradient);
    auto const elems_to_nodes = this->get_elems_to_nodes();
    auto const nodes_to_v = this->sim.get(this->sim.velocity);
    auto const dt = this->sim.dt;
    auto functor = OMEGA_H_LAMBDA(int const point) {
      auto const elem = point / Elem::points;
      auto const elem_nodes = getnodes<Elem>(elems_to_nodes, elem);
      auto const v = getvecs<Elem>(nodes_to_v, elem_nodes);
      auto const grads = getgrads<Elem>(points_to_grad, point);
      auto const grad_v = grad<Elem>(grads, v);
      auto const I = identity_matrix<Elem::dim, Elem::dim>();
      auto const grad_u = dt * grad_v;
      auto const incr_F = I + grad_u;
      OMEGA_H_CHECK(determinant(incr_F) > 0.0);
      auto const old_F = getfull<Elem>(points_to_F, point);
      OMEGA_H_CHECK(determinant(old_F) > 0.0);
      auto const new_F = incr_F * old_F;
      OMEGA_H_CHECK(determinant(new_F) > 0.0);
      setfull<Elem>(points_to_F, point, new_F);
    };
    parallel_for(this->points(), std::move(functor));
  }
};

template <class Elem>
ModelBase* deformation_gradient_factory(
    Simulation& sim, std::string const&, Omega_h::InputMap&) {
  return new DeformationGradient<Elem>(sim);
}

#define LGR_EXPL_INST(Elem)                                                    \
  template ModelBase* deformation_gradient_factory<Elem>(                      \
      Simulation&, std::string const&, Omega_h::InputMap&);
LGR_EXPL_INST_ELEMS
#undef LGR_EXPL_INST

}  // namespace lgr
