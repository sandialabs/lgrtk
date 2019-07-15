#ifndef SIMPLEX_ELECTROMECHANICS_HPP
#define SIMPLEX_ELECTROMECHANICS_HPP

#include "plato/Simplex.hpp"
#include "plato/PlatoStaticsTypes.hpp"

namespace Plato
{

/******************************************************************************/
/*! Base class for simplex-based electromechanics
*/
/******************************************************************************/
template<Plato::OrdinalType SpaceDim, Plato::OrdinalType NumControls = 1>
class SimplexElectromechanics : public Plato::Simplex<SpaceDim>
{
  public:
    using Plato::Simplex<SpaceDim>::mNumNodesPerCell;
    using Plato::Simplex<SpaceDim>::mNumSpatialDims;

    static constexpr Plato::OrdinalType mNumVoigtTerms   = (SpaceDim == 3) ? 6 :
                                             ((SpaceDim == 2) ? 3 :
                                            (((SpaceDim == 1) ? 1 : 0)));
    static constexpr Plato::OrdinalType mNumDofsPerNode  = SpaceDim + 1;
    static constexpr Plato::OrdinalType mNumDofsPerCell  = mNumDofsPerNode*mNumNodesPerCell;

    static constexpr Plato::OrdinalType mNumControl = NumControls;
};

} // namespace Plato

#endif
