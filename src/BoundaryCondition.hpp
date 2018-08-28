#ifndef LGR_BOUNDARY_CONDITION_HPP
#define LGR_BOUNDARY_CONDITION_HPP

#include "VectorContribution.hpp"
#include "MeshIO.hpp"

namespace lgr {

template <int SpatialDim>
class VectorBoundaryCondition :
  public VectorContribution<SpatialDim> {
  public:
    using typename VectorContribution<SpatialDim>::geom_array_type;
    using typename VectorContribution<SpatialDim>::node_coords_type;
    NodeSet ns;
    int component;
    Kokkos::View<const Scalar*> values;
    std::string name;
    VectorBoundaryCondition(
        std::string const& name_in, Teuchos::ParameterList& pl);
    virtual void add_to(geom_array_type field) const override final;
    virtual ~VectorBoundaryCondition() {};
};

template <int SpatialDim>
class FixedVectorBoundaryCondition :
  public VectorBoundaryCondition<SpatialDim> {
  public:
    using typename VectorBoundaryCondition<SpatialDim>::node_coords_type;
    Scalar value;
    FixedVectorBoundaryCondition(
        std::string const& name_in, Teuchos::ParameterList& pl);
    virtual void update(
        const Omega_h::MeshSets &,
        const Scalar,
        const node_coords_type) override final;
    virtual ~FixedVectorBoundaryCondition() {};
};

template <int SpatialDim>
class FunctionVectorBoundaryCondition :
  public VectorBoundaryCondition<SpatialDim> {
  public:
    using typename VectorBoundaryCondition<SpatialDim>::node_coords_type;
    std::string expr;
    FunctionVectorBoundaryCondition(
        std::string const& name_in, Teuchos::ParameterList& pl);
    virtual void update(
        const Omega_h::MeshSets &,
        const Scalar,
        const node_coords_type) override final;
    virtual ~FunctionVectorBoundaryCondition() {};
};

template <int SpatialDim>
void mark_fixed_velocity(
    VectorContributions<SpatialDim> const& accel_contribs,
    MeshIO& mesh_io);

template <int SpatialDim>
void load_boundary_conditions(
    VectorContributions<SpatialDim>& contribs,
    Teuchos::ParameterList& pl);

#define LGR_EXPL_DECL_INST(SpatialDim) \
extern template class VectorBoundaryCondition<SpatialDim>; \
extern template class FixedVectorBoundaryCondition<SpatialDim>; \
extern template class FunctionVectorBoundaryCondition<SpatialDim>; \
extern template \
void mark_fixed_velocity( \
    VectorContributions<SpatialDim> const& accel_contribs, \
    MeshIO& mesh_io); \
extern template \
void load_boundary_conditions( \
    VectorContributions<SpatialDim>& contribs, \
    Teuchos::ParameterList& pl);
LGR_EXPL_DECL_INST(3)
LGR_EXPL_DECL_INST(2 )
#undef LGR_EXPL_DECL_INST


}

#endif
