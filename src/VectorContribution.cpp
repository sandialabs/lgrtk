#include <ErrorHandling.hpp>
#include "VectorContribution.hpp"

namespace lgr {

NodeSet::NodeSet(Teuchos::ParameterList& pl) {
  if (pl.isType<std::string>("Sides")) {
    name = pl.get<std::string>("Sides");
  } else {
    name = pl.get<std::string>("Node Set");
  }
}

static Omega_h::LOs get_set_nodes(
    std::string const& node_set_name,
    Omega_h::MeshSets const& mesh_sets) {
  auto& node_sets = mesh_sets[Omega_h::NODE_SET];
  auto node_set_it = node_sets.find(node_set_name);
  LGR_THROW_IF(node_set_it == node_sets.end(),
      "Node set " << node_set_name << " not found!\n");
  return node_set_it->second;
}

void NodeSet::update(const Omega_h::MeshSets& mesh_sets) {
  nodes = get_set_nodes(name, mesh_sets);
}

template <int SpatialDim>
VectorContribution<SpatialDim>::~VectorContribution() {
}

template <int SpatialDim>
void VectorContributions<SpatialDim>::add_to(
    geom_array_type field) const {
  for (auto& contrib : contribs) {
    contrib->add_to(field);
  }
}

template <int SpatialDim>
void VectorContributions<SpatialDim>::update(
    const Omega_h::MeshSets& mesh_sets,
    const Scalar time,
    const node_coords_type coords) {
  for (auto& contrib : contribs) {
    contrib->update(mesh_sets, time, coords);
  }
}

template <int SpatialDim>
void VectorContributions<SpatialDim>::add(VectorContribution<SpatialDim>* contrib) {
  contribs.emplace_back(contrib);
}

#define LGR_EXPL_DECL(SpatialDim) \
template class VectorContribution<SpatialDim>; \
template class VectorContributions<SpatialDim>;
LGR_EXPL_DECL(3)
LGR_EXPL_DECL(2)
#undef LGR_EXPL_DECL

}
