/*
 * SimplexStructuralDynamics.hpp
 *
 *  Created on: Apr 21, 2018
 */

#ifndef SIMPLEXSTRUCTURALDYNAMICS_HPP_
#define SIMPLEXSTRUCTURALDYNAMICS_HPP_

#include "plato/Simplex.hpp"
#include "plato/PlatoStaticsTypes.hpp"

namespace Plato
{

template<Plato::OrdinalType SpaceDim, Plato::OrdinalType NumControls = 1>
class SimplexStructuralDynamics : public Plato::Simplex<SpaceDim>
{
public:
    using Plato::Simplex<SpaceDim>::mNumNodesPerCell;
    using Plato::Simplex<SpaceDim>::mNumSpatialDims;

    static constexpr Plato::OrdinalType mComplexSpaceDim = 2;
    static constexpr Plato::OrdinalType mNumVoigtTerms = (SpaceDim == 3) ? 6 : ((SpaceDim == 2) ? 3: (((SpaceDim == 1) ? 1: 0)));
    static constexpr Plato::OrdinalType mNumDofsPerNode = mComplexSpaceDim * SpaceDim;
    static constexpr Plato::OrdinalType mNumDofsPerCell = mNumDofsPerNode * mNumNodesPerCell;

    static constexpr Plato::OrdinalType mNumControl = NumControls;

    static constexpr Plato::OrdinalType mNumNSPerNode = 0;

    static constexpr Plato::OrdinalType mNumLocalDofsPerCell = 0;

};
// class SimplexStructuralDynamics

} // namespace Plato

#endif /* SIMPLEXSTRUCTURALDYNAMICS_HPP_ */
