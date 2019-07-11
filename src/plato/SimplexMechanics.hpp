#ifndef SIMPLEX_MECHANICS_HPP
#define SIMPLEX_MECHANICS_HPP

#include "plato/Simplex.hpp"
#include "plato/PlatoStaticsTypes.hpp"

namespace Plato
{

/******************************************************************************/
/*! Base class for simplex-based mechanics
*/
/******************************************************************************/
template<Plato::OrdinalType SpaceDim, Plato::OrdinalType NumControls = 1>
class SimplexMechanics : public Plato::Simplex<SpaceDim>
{
  public:
    using Plato::Simplex<SpaceDim>::m_numNodesPerCell;
    using Plato::Simplex<SpaceDim>::m_numSpatialDims;

    static constexpr Plato::OrdinalType m_numVoigtTerms   = (SpaceDim == 3) ? 6 :
                                             ((SpaceDim == 2) ? 3 :
                                            (((SpaceDim == 1) ? 1 : 0)));
    static constexpr Plato::OrdinalType m_numDofsPerNode  = SpaceDim;
    static constexpr Plato::OrdinalType m_numDofsPerCell  = m_numDofsPerNode*m_numNodesPerCell;

    static constexpr Plato::OrdinalType m_numControl = NumControls;

    static constexpr Plato::OrdinalType m_numNSPerNode = 0;

};

} // namespace Plato

#endif
