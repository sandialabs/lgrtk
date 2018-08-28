#include "ContactForce.hpp"

#include <LGRLambda.hpp>
#include <ErrorHandling.hpp>
#include <FieldDB.hpp>
#include <Omega_h_map.hpp>
#include <Omega_h_mesh.hpp>

namespace lgr {

template <int SpatialDim>
void compute_contact_forces(VectorContributions<SpatialDim>& forces,
                            Teuchos::ParameterList&          params) {
  for (auto i = params.begin(); i != params.end(); ++i) {
    auto& name = params.name(i);

    auto& entry = params.entry(i);

    LGR_THROW_IF(entry.isList() == false,
                   "Parameter "
                       << name
                       << " in Contact block not valid.\nExpect lists only.");

    auto& sublist = params.sublist(name);

    auto& type = sublist.get<std::string>("Type");

    if (starts_with(type, "Penalty")) {
      forces.add(new PenaltyContactForce<SpatialDim>(name, sublist));
    } else {
      LGR_THROW_IF(true, "Invalid Contact type " << type);
    }
  }
  return;
}

template <int SpatialDim>
ContactForce<SpatialDim>::ContactForce(std::string const&      name,
                                       Teuchos::ParameterList& params)
    : name_(name), node_set_(params) {
  if (params.isType<Scalar>("Gap Length")) {
    gap_length_ = params.get<Scalar>("Gap Length");
  }
  return;
}

template <int SpatialDim>
ContactForce<SpatialDim>::~ContactForce() {
  return;
}

template <int SpatialDim>
void ContactForce<SpatialDim>::add_to(geom_array_type field) const {
  auto nodes_local = this->node_set_.nodes;
  auto values_local = this->value_;
  auto f = LAMBDA_EXPRESSION(int set_node) {
    auto node = nodes_local[set_node];
    for (auto d = 0; d < SpatialDim; ++d) {
      field(node, d) += values_local(set_node, d);
    }
  };
  Kokkos::parallel_for(nodes_local.size(), f);
}

template <int SpatialDim>
PenaltyContactForce<SpatialDim>::PenaltyContactForce(
    std::string const& name, Teuchos::ParameterList& params)
    : ContactForce<SpatialDim>(name, params) {
  if (params.isType<Scalar>("Penalty Coefficient")) {
    penalty_coefficient_ = params.get<Scalar>("Penalty Coefficient");
  }
  return;
}

template <int SpatialDim>
PenaltyContactForce<SpatialDim>::~PenaltyContactForce() {
  return;
}

template <int SpatialDim>
Omega_h::LOs getBoundary(Omega_h::Mesh& mesh) {
  int const space_dim = mesh.dim();

  LGR_THROW_IF(space_dim != SpatialDim, "Inconsistent dimensions!\n");

  int const boundary_dim = SpatialDim - 1;

  Omega_h::Read<Omega_h::I8> interior_marks =
      Omega_h::mark_by_class_dim(&mesh, boundary_dim, space_dim);

  Omega_h::Read<Omega_h::I8> boundary_marks =
      Omega_h::invert_marks(interior_marks);

  Omega_h::LOs local_ordinals = Omega_h::collect_marked(boundary_marks);

  return local_ordinals;
}

template <int SpatialDim>
void PenaltyContactForce<SpatialDim>::update(Omega_h::MeshSets const& mesh_sets,
                                             Scalar const,
                                             node_coords_type const) {
  auto&& node_set = this->node_set_;

  node_set.update(mesh_sets);

  auto set_nodes = node_set.nodes;

  auto const num_nodes = set_nodes.size();

  Kokkos::View<Scalar*[SpatialDim], MemSpace>
  forces("penalty_forces", num_nodes);

  auto const gap = this->gap_length_;

  auto const K = penalty_coefficient_;

  constexpr int k = SpatialDim - 1;

  using F = Fields<SpatialDim>;
  auto nodal_mass = NodalMass<F>();
  auto spatial_coordinates = F::getGeomFromSA(Coordinates<F>(), 0);

  auto enforce = OMEGA_H_LAMBDA(int set_node) {
    auto node = set_nodes[set_node];
    auto z = spatial_coordinates(node, k);
    if (z < gap) {
      auto const mass = nodal_mass(node);
      forces(set_node, k) = - K * mass * (gap - z);
    }
  };
  Kokkos::parallel_for(num_nodes, enforce);

  this->value_ = forces;

  return;
}

#define LGR_EXPL_DECL(SpatialDim)                                            \
  template void compute_contact_forces(                                        \
      VectorContributions<SpatialDim>& forces,                                 \
      Teuchos::ParameterList&          params);
LGR_EXPL_DECL(3)
LGR_EXPL_DECL(2)
#undef LGR_EXPL_DECL

}  // namespace lgr
