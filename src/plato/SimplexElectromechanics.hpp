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
class SimplexElectromechanics : public Simplex<SpaceDim>
{
  public:
    using Simplex<SpaceDim>::m_numNodesPerCell;
    using Simplex<SpaceDim>::m_numSpatialDims;

    static constexpr Plato::OrdinalType m_numVoigtTerms   = (SpaceDim == 3) ? 6 :
                                             ((SpaceDim == 2) ? 3 :
                                            (((SpaceDim == 1) ? 1 : 0)));
    static constexpr Plato::OrdinalType m_numDofsPerNode  = SpaceDim + 1;
    static constexpr Plato::OrdinalType m_numDofsPerCell  = m_numDofsPerNode*m_numNodesPerCell;

    static constexpr Plato::OrdinalType m_numControl = NumControls;
};

} // namespace Plato

#endif
