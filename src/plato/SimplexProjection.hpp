#ifndef SIMPLEX_PROJECTION_HPP
#define SIMPLEX_PROJECTION_HPP

#include "plato/Simplex.hpp"
#include "plato/PlatoStaticsTypes.hpp"

namespace Plato
{

/******************************************************************************/
/*! Base class for simplex-based projection
*/
/******************************************************************************/
template<
  Plato::OrdinalType SpaceDim,
  Plato::OrdinalType TotalDofs = SpaceDim,
  Plato::OrdinalType ProjectionDofOffset = 0,
  Plato::OrdinalType NumProjectionDof = 1,
  Plato::OrdinalType NumControls = 1>
class SimplexProjection : public Plato::Simplex<SpaceDim>
{
  public:
    using Plato::Simplex<SpaceDim>::mNumNodesPerCell;
    using Plato::Simplex<SpaceDim>::mNumSpatialDims;

    static constexpr Plato::OrdinalType mPDofOffset     = SpaceDim;
    static constexpr Plato::OrdinalType mTotalDofs      = TotalDofs;
    static constexpr Plato::OrdinalType mProjectionDof  = ProjectionDofOffset;
    static constexpr Plato::OrdinalType mNumDofsPerNode = SpaceDim;
    static constexpr Plato::OrdinalType mNumDofsPerCell = mNumDofsPerNode*mNumNodesPerCell;
    static constexpr Plato::OrdinalType mNumControl     = NumControls;

    // this physics can be used with VMS functionality in PA.  The
    // following defines the nodal state attributes required by VMS
    //
    static constexpr Plato::OrdinalType mNumNSPerNode    = NumProjectionDof;
    static constexpr Plato::OrdinalType mNumNSPerCell    = mNumNSPerNode*mNumNodesPerCell;

    static constexpr Plato::OrdinalType mNumLocalDofsPerCell = 0;
};

} // namespace Plato

#endif
