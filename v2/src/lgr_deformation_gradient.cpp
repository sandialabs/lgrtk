#include <lgr_deformation_gradient.hpp>
#include <lgr_model.hpp>
#include <lgr_simulation.hpp>
#include <lgr_for.hpp>
#include <Omega_h_print.hpp>

namespace lgr {

template <class Elem>
struct DeformationGradient : public Model<Elem> {
  FieldIndex deformation_gradient;
  DeformationGradient(Simulation& sim_in)
    :Model<Elem>(sim_in, sim_in.fields[sim_in.fields.find("deformation gradient")].class_names)
    ,deformation_gradient(sim_in.fields.find("deformation gradient"))
  {
  }
  std::uint64_t exec_stages() override final { return AT_FIELD_UPDATE; }
  char const* name() override final { return "deformation gradient"; }
  void at_field_update() override final {
    auto points_to_grad = this->points_get(this->sim.gradient);
    auto points_to_F = this->points_getset(this->deformation_gradient);
    auto elems_to_nodes = this->get_elems_to_nodes();
    auto nodes_to_v = this->sim.get(this->sim.velocity);
    auto dt = this->sim.dt;
    auto functor = OMEGA_H_LAMBDA(int point) {
      auto elem = point / Elem::points;
      auto elem_nodes = getnodes<Elem>(elems_to_nodes, elem);
      auto v = getvecs<Elem>(nodes_to_v, elem_nodes);
      auto dN_dxnp1 = getgrads<Elem>(points_to_grad, point);
      auto dv_dxnp1 = grad<Elem>(dN_dxnp1, v);
      auto I = identity_matrix<Elem::dim, Elem::dim>();
      auto dxn_dxnp1 = I - dt * dv_dxnp1;
      auto dxnp1_dxn = invert(dxn_dxnp1);
      auto dxn_dX = getfull<Elem>(points_to_F, point);
      if (!(determinant(dxn_dX) > 0.0)) {
        std::cerr << "got old F = " << dxn_dX << '\n';
      }
      OMEGA_H_CHECK(determinant(dxn_dX) > 0.0);
      OMEGA_H_CHECK(determinant(dxnp1_dxn) > 0.0);
      auto dxnp1_dX = dxn_dX * dxnp1_dxn;
      OMEGA_H_CHECK(determinant(dxnp1_dX) > 0.0);
      setfull<Elem>(points_to_F, point, dxnp1_dX);
    };
    parallel_for("deformation gradient kernel", this->points(), std::move(functor));
  }
};

template <class Elem>
ModelBase* deformation_gradient_factory(Simulation& sim, std::string const&, Teuchos::ParameterList&) {
  return new DeformationGradient<Elem>(sim);
}

#define LGR_EXPL_INST(Elem) \
template ModelBase* deformation_gradient_factory<Elem>(Simulation&, std::string const&, Teuchos::ParameterList&);
LGR_EXPL_INST_ELEMS
#undef LGR_EXPL_INST

}
