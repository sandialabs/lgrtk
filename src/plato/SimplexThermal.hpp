#ifndef SIMPLEX_THERMAL_HPP
#define SIMPLEX_THERMAL_HPP

#include "plato/Simplex.hpp"

/******************************************************************************/
/*! Base class for simplex-based thermal
*/
/******************************************************************************/
template<Plato::OrdinalType SpaceDim>
class SimplexThermal : public Simplex<SpaceDim>
{ 
  public:
    using Simplex<SpaceDim>::m_numNodesPerCell;
    using Simplex<SpaceDim>::m_numSpatialDims;

    static constexpr Plato::OrdinalType m_numDofsPerNode  = 1;
    static constexpr Plato::OrdinalType m_numDofsPerCell  = m_numDofsPerNode*m_numNodesPerCell;

    static constexpr Plato::OrdinalType m_numControl = 1;  // for now, only one control field allowed.

};
#endif
