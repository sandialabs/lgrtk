#ifndef SIMPLEX_PHYSICS_HPP
#define SIMPLEX_PHYSICS_HPP

#include "plato/PlatoStaticsTypes.hpp"

/******************************************************************************/
/*! Base class for simplex-based mechanics
*/
/******************************************************************************/
template<Plato::OrdinalType SpaceDim>
class Simplex
{
  public:
    static constexpr Plato::OrdinalType m_numSpatialDims  = SpaceDim;
    static constexpr Plato::OrdinalType m_numNodesPerCell = SpaceDim+1;
};
#endif
