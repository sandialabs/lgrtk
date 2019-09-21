#pragma once

#include "plato/PlatoStaticsTypes.hpp"

namespace Plato
{

/******************************************************************************/
/*! Base class for simplex-based mechanics
*/
/******************************************************************************/
template<Plato::OrdinalType SpaceDim>
class Simplex
{
  public:
    static constexpr Plato::OrdinalType mNumSpatialDims  = SpaceDim;
    static constexpr Plato::OrdinalType mNumNodesPerCell = SpaceDim+1;
};
// class Simplex

} // namespace Plato
