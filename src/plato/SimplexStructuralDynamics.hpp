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
    using Plato::Simplex<SpaceDim>::m_numNodesPerCell;
    using Plato::Simplex<SpaceDim>::m_numSpatialDims;

    static constexpr Plato::OrdinalType mComplexSpaceDim = 2;
    static constexpr Plato::OrdinalType m_numVoigtTerms = (SpaceDim == 3) ? 6 : ((SpaceDim == 2) ? 3: (((SpaceDim == 1) ? 1: 0)));
    static constexpr Plato::OrdinalType m_numDofsPerNode = mComplexSpaceDim * SpaceDim;
    static constexpr Plato::OrdinalType m_numDofsPerCell = m_numDofsPerNode * m_numNodesPerCell;

    static constexpr Plato::OrdinalType m_numControl = NumControls;

    static constexpr Plato::OrdinalType m_numNSPerNode = 0;

};
// class SimplexStructuralDynamics

} // namespace Plato

#endif /* SIMPLEXSTRUCTURALDYNAMICS_HPP_ */
