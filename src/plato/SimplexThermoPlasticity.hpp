#pragma once

#include "plato/Simplex.hpp"
#include "plato/PlatoStaticsTypes.hpp"

namespace Plato
{

template<Plato::OrdinalType SpaceDim, Plato::OrdinalType NumControls = 1>
class SimplexThermoPlasticity : public Plato::Simplex<SpaceDim>
{
public:
    using Plato::Simplex<SpaceDim>::mNumNodesPerCell;
    using Plato::Simplex<SpaceDim>::mNumSpatialDims;

    static constexpr Plato::OrdinalType mNumVoigtTerms = (SpaceDim == 3) ? 6 : ((SpaceDim == 2) ? 3: (((SpaceDim == 1) ? 1: 0)));
    static constexpr Plato::OrdinalType mNumDofsPerNode = SpaceDim + 2; // displacement + pressure + temperature
    static constexpr Plato::OrdinalType mNumDofsPerCell = mNumDofsPerNode * mNumNodesPerCell;

    static constexpr Plato::OrdinalType mNumLocalDofsPerCell = (SpaceDim == 3) ? 14 : ((SpaceDim == 2) ? 8: (((SpaceDim == 1) ? 4: 0)));

    static constexpr Plato::OrdinalType mNumControl = NumControls;
};
// class SimplexPlasticity

} // namespace Plato
