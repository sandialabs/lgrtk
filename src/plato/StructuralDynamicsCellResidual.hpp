/*
 * StructuralDynamicsCellResidual.hpp
 *
 *  Created on: Apr 23, 2018
 */

#ifndef STRUCTURALDYNAMICSCELLRESIDUAL_HPP_
#define STRUCTURALDYNAMICSCELLRESIDUAL_HPP_

#include <cassert>

#include "plato/PlatoStaticsTypes.hpp"

namespace Plato
{

/****************************************************************************//**
 *
 * \brief Inline function used to compute the structural dynamics residual.
 *
 *   Compute the structural dynamics residual at a given cell (i.e. element).
 *   The structural dynamics residual is given by
 *
 *     \f$\mathbf{R}\equiv\mathbf{M}\bar{\{\ddot{\mathbf{D}}\}} + \mathbf{C}
 *     \bar{\{\dot{\mathbf{D}}\}} + \mathbf{K}\bar{\{\mathbf{D}\}} +
 *     \bar{\{\mathbf{F}^{ext}\}} = 0\f$,
 *
 *   where \f$\mathbf{M}\f$ is the mass matrix, \f$\mathbf{M}\f$ is the damping
 *   matrix, \f$\mathbf{M}\f$ is the stiffness matrix, \f$\bar{\{\ddot{\mathbf{D}}\}}\f$
 *   is the complex acceleration vector, \f$\bar{\{\dot{\mathbf{D}}\}}\f$ is
 *   the complex velocity vector, \f$\bar{\{\ddot{\mathbf{D}}\}}\f$ is the
 *   complex displacement vector and \f$\bar{\{\mathbf{F}^{ext}\}}\f$ is the
 *   complex external force vector.
 *
********************************************************************************/
template<Plato::OrdinalType NumDofsPerNode, typename ResultScalarType>
KOKKOS_INLINE_FUNCTION void
structural_dynamics_cell_residual(const Plato::OrdinalType & aCellOrdinal,
                                  const Plato::Scalar & aOmegaTimesOmega,
                                  const Plato::ScalarMultiVectorT<ResultScalarType> & aElasticForces,
                                  const Plato::ScalarMultiVectorT<ResultScalarType> & aDampingForces,
                                  const Plato::ScalarMultiVectorT<ResultScalarType> & aInertialForces,
                                  const Plato::ScalarMultiVectorT<ResultScalarType> & aResidual)
{
    assert(aElasticForces.size() == aResidual.size());
    assert(aElasticForces.size() == aDampingForces.size());
    assert(aElasticForces.size() == aInertialForces.size());
    assert(aCellOrdinal >= static_cast<Plato::Scalar>(0.0));
    assert(aOmegaTimesOmega >= static_cast<Plato::Scalar>(0.0));

    for(Plato::OrdinalType tIndex = 0; tIndex < NumDofsPerNode; tIndex++)
    {
        aResidual(aCellOrdinal, tIndex) += aElasticForces(aCellOrdinal, tIndex) + aDampingForces(aCellOrdinal, tIndex)
                - (aOmegaTimesOmega * aInertialForces(aCellOrdinal, tIndex));
    }
}

}// namespace Plato

#endif /* STRUCTURALDYNAMICSCELLRESIDUAL_HPP_ */
