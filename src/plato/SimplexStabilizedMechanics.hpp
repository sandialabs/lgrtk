#ifndef SIMPLEX_STABILIZED_MECHANICS_HPP
#define SIMPLEX_STABILIZED_MECHANICS_HPP

#include "plato/Simplex.hpp"
#include "plato/PlatoStaticsTypes.hpp"

namespace Plato
{

/******************************************************************************/
/*! Base class for simplex-based two-field Mechanics
*/
/******************************************************************************/
template<Plato::OrdinalType SpaceDim, Plato::OrdinalType NumControls = 1>
class SimplexStabilizedMechanics : public Plato::Simplex<SpaceDim>
{
  public:
    using Plato::Simplex<SpaceDim>::mNumNodesPerCell;
    using Plato::Simplex<SpaceDim>::mNumSpatialDims;

    static constexpr Plato::OrdinalType mNumVoigtTerms   = (SpaceDim == 3) ? 6 :
                                             ((SpaceDim == 2) ? 3 :
                                            (((SpaceDim == 1) ? 1 : 0)));

    // degree-of-freedom attributes
    //
    static constexpr Plato::OrdinalType mPDofOffset      = SpaceDim;
    static constexpr Plato::OrdinalType mNumDofsPerNode  = SpaceDim + 1;
    static constexpr Plato::OrdinalType mNumDofsPerCell  = mNumDofsPerNode*mNumNodesPerCell;
    static constexpr Plato::OrdinalType mNumControl      = NumControls;

    // this physics can be used with VMS functionality in PA.  The
    // following defines the nodal state attributes required by VMS
    //
    static constexpr Plato::OrdinalType mNumNSPerNode    = SpaceDim;
    static constexpr Plato::OrdinalType mNumNSPerCell    = mNumNSPerNode*mNumNodesPerCell;

    static constexpr Plato::OrdinalType mNumLocalDofsPerCell = 0;
};

} // namespace Plato

#endif
