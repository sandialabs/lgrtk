#ifndef SIMPLEX_THERMAL_HPP
#define SIMPLEX_THERMAL_HPP

#include "plato/Simplex.hpp"

namespace Plato
{

/******************************************************************************/
/*! Base class for simplex-based thermal
*/
/******************************************************************************/
template<Plato::OrdinalType SpaceDim>
class SimplexThermal : public Plato::Simplex<SpaceDim>
{ 
  public:
    using Plato::Simplex<SpaceDim>::m_numNodesPerCell;
    using Plato::Simplex<SpaceDim>::m_numSpatialDims;

    static constexpr Plato::OrdinalType m_numDofsPerNode  = 1;
    static constexpr Plato::OrdinalType m_numDofsPerCell  = m_numDofsPerNode*m_numNodesPerCell;

    static constexpr Plato::OrdinalType m_numControl = 1;  // for now, only one control field allowed.

};
// class SimplexThermal

} // namespace Plato

#endif
