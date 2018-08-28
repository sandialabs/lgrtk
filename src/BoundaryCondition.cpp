#include "BoundaryCondition.hpp"

#include <ErrorHandling.hpp>
#include <LGRLambda.hpp>
#include <Omega_h_expr.hpp>

namespace lgr {

template <int SpatialDim>
VectorBoundaryCondition<SpatialDim>::VectorBoundaryCondition(
    std::string const& name_in, Teuchos::ParameterList& pl):
  ns(pl),
  name(name_in) {
  if (pl.isType<int>("Index")) {
    component = pl.get<int>("Index");
  } else {
    component = pl.get<int>("Component");
  }
}

template <int SpatialDim>
void VectorBoundaryCondition<SpatialDim>::add_to(
    geom_array_type field) const {
  auto component_local = this->component;
  auto nodes_local = this->ns.nodes;
  auto values_local = this->values;
  auto f = LAMBDA_EXPRESSION(int set_node) {
    auto node = nodes_local[set_node];
    field(node, component_local) = values_local[set_node];
  };
  Kokkos::parallel_for(nodes_local.size(), f);
}

template <int SpatialDim>
FixedVectorBoundaryCondition<SpatialDim>::FixedVectorBoundaryCondition(
    std::string const& name_in, Teuchos::ParameterList& pl):
  VectorBoundaryCondition<SpatialDim>(name_in, pl) {
  value = pl.get<double>("Value");
}

template <int SpatialDim>
void FixedVectorBoundaryCondition<SpatialDim>::update(
        const Omega_h::MeshSets& mesh_sets,
        const Scalar,
        const node_coords_type) {
  this->ns.update(mesh_sets);
  Kokkos::View<Scalar*, MemSpace> values_w("fixed_values", this->ns.nodes.size());
  Kokkos::deep_copy(values_w, value);
  this->values = values_w;
}

template <int SpatialDim>
static Omega_h::Reals evaluate_expression_at_set_nodes(
    std::string const& expr,
    std::string const& name,
    double time,
    typename Fields<SpatialDim>::geom_array_type coords,
    Omega_h::LOs set_nodes) {
  auto nset_nodes = set_nodes.size();
  Omega_h::Few<Omega_h::Write<Omega_h::Real>, SpatialDim> coords_w;
  for (int d = 0; d < SpatialDim; ++d) {
    coords_w[d] = Omega_h::Write<Omega_h::Real>(nset_nodes);
  }
  auto prepare = OMEGA_H_LAMBDA(int set_node) {
    auto node = set_nodes[set_node];
    for (int d = 0; d < SpatialDim; ++d) {
      coords_w[d][set_node] = coords(node, d);
    }
  };
  Kokkos::parallel_for(nset_nodes, prepare);
  Omega_h::ExprReader reader(nset_nodes, SpatialDim);
  const char* const dim_names[3] = {"x", "y", "z"};
  for (int d = 0; d < SpatialDim; ++d) {
    reader.register_variable(dim_names[d],
        Teuchos::any(Omega_h::Reals(coords_w[d])));
  }
  reader.register_variable("t", Teuchos::any(Omega_h::Real(time)));
  Teuchos::any result_any;
  reader.read_string(result_any, expr, name);
  reader.repeat(result_any);
  return Teuchos::any_cast<Omega_h::Reals>(result_any);
}

template <int SpatialDim>
FunctionVectorBoundaryCondition<SpatialDim>::FunctionVectorBoundaryCondition(
    std::string const& name_in, Teuchos::ParameterList& pl):
  VectorBoundaryCondition<SpatialDim>(name_in, pl) {
  expr = pl.get<std::string>("Value");
}

template <int SpatialDim>
void FunctionVectorBoundaryCondition<SpatialDim>::update(
    const Omega_h::MeshSets& mesh_sets,
    const Scalar time,
    const node_coords_type coords) {
  this->ns.update(mesh_sets);
  auto osh_values =
    evaluate_expression_at_set_nodes<SpatialDim>(
        expr, this->name, time, coords, this->ns.nodes);
  this->values = osh_values.view();
}

template <int SpatialDim>
void mark_fixed_velocity(
    VectorContributions<SpatialDim> const& accel_contribs,
    MeshIO& mesh_io) {
  for (auto& contrib : accel_contribs.contribs) {
    VectorContribution<SpatialDim> const* contrib_ptr =
      contrib.get();
    auto bc_ptr =
      dynamic_cast<VectorBoundaryCondition<SpatialDim> const*>(contrib_ptr);
    if (bc_ptr) {
      mesh_io.markFixedVelocity(bc_ptr->ns.name,
          bc_ptr->component);
    }
  }
}

template <int SpatialDim>
void load_boundary_conditions(
    VectorContributions<SpatialDim>& contribs,
    Teuchos::ParameterList& pl) {
  for (auto i = pl.begin(); i != pl.end(); ++i) {
    auto& entry = pl.entry(i);
    auto& name = pl.name(i);

    LGR_THROW_IF(
        !entry.isList(),
        "Parameter " << name
        << " in Boundary Conditions block not valid.\n"
        "Expect lists only.");

    auto& sublist = pl.sublist(name);
    auto& type = sublist.get<std::string>("Type");
    if (starts_with(type, "Zero")) {
      sublist.set<double>("Value", 0.0);
      contribs.add(
          new FixedVectorBoundaryCondition<SpatialDim>(
            name, sublist));
    } else if (starts_with(type, "Fixed")) {
      contribs.add(
          new FixedVectorBoundaryCondition<SpatialDim>(
            name, sublist));
    } else if (starts_with(type, "Function")) {
      contribs.add(
          new FunctionVectorBoundaryCondition<SpatialDim>(
            name, sublist));
    } else {
      LGR_THROW_IF(true,
          "Invalid Boundary Condition type "
          << type);
    }
  }
}

#define LGR_EXPL_DECL(SpatialDim) \
template class VectorBoundaryCondition<SpatialDim>; \
template class FixedVectorBoundaryCondition<SpatialDim>; \
template class FunctionVectorBoundaryCondition<SpatialDim>; \
template \
void mark_fixed_velocity( \
    VectorContributions<SpatialDim> const& accel_contribs, \
    MeshIO& mesh_io); \
template \
void load_boundary_conditions( \
    VectorContributions<SpatialDim>& contribs, \
    Teuchos::ParameterList& pl);
LGR_EXPL_DECL(3)
LGR_EXPL_DECL(2)
#undef LGR_EXPL_DECL

}
