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
    using Plato::Simplex<SpaceDim>::mNumNodesPerCell;
    using Plato::Simplex<SpaceDim>::mNumSpatialDims;

    static constexpr Plato::OrdinalType mNumDofsPerNode  = 1;
    static constexpr Plato::OrdinalType mNumDofsPerCell  = mNumDofsPerNode*mNumNodesPerCell;

    static constexpr Plato::OrdinalType mNumControl = 1;  // for now, only one control field allowed.

    static constexpr Plato::OrdinalType mNumNSPerNode = 0;

    static constexpr Plato::OrdinalType mNumLocalDofsPerCell = 0;

};
// class SimplexThermal

} // namespace Plato

#endif
