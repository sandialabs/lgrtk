#ifndef LGR_VECTOR_CONTRIBUTION_HPP
#define LGR_VECTOR_CONTRIBUTION_HPP

#include <vector>
#include <memory>
#include <Fields.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Omega_h_assoc.hpp>

namespace lgr {

inline bool
starts_with(std::string const& s, std::string const& prefix)
{
  if (s.length() < prefix.length()) return false;
  return 0 == s.compare(0, prefix.length(), prefix);
}

struct NodeSet {
  std::string name;
  Omega_h::LOs nodes;
  NodeSet(Teuchos::ParameterList& pl);
  void update(const Omega_h::MeshSets& mesh_sets);
};

template <int SpatialDim>
class VectorContribution {
  public:
    using geom_array_type = typename Fields<SpatialDim>::geom_array_type;
    using node_coords_type = typename Fields<SpatialDim>::node_coords_type;
    virtual void add_to(geom_array_type field) const = 0;
    virtual void update(
        const Omega_h::MeshSets &,
        const Scalar,
        const node_coords_type) = 0;
    virtual ~VectorContribution();
};

template <int SpatialDim>
class VectorContributions {
  public:
    using geom_array_type = typename Fields<SpatialDim>::geom_array_type;
    using node_coords_type = typename Fields<SpatialDim>::node_coords_type;
    std::vector<std::unique_ptr<VectorContribution<SpatialDim>>> contribs;
    void add_to(geom_array_type field) const;
    void update(
        const Omega_h::MeshSets &,
        const Scalar,
        const node_coords_type);
    void add(VectorContribution<SpatialDim>* contrib);
};
#define LGR_EXPL_DECL_INST(SpatialDim) \
extern template class VectorContribution<SpatialDim>; \
extern template class VectorContributions<SpatialDim>;
LGR_EXPL_DECL_INST(3)
LGR_EXPL_DECL_INST(2)
#undef LGR_EXPL_DECL_INST

}

#endif
