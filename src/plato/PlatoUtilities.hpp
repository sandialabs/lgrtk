/*
 * PlatoUtilities.hpp
 *
 *  Created on: Aug 8, 2018
 */

#ifndef SRC_PLATO_PLATOUTILITIES_HPP_
#define SRC_PLATO_PLATOUTILITIES_HPP_

#include <Omega_h_array.hpp>

#include "plato/PlatoStaticsTypes.hpp"

namespace Plato
{

template<const Plato::OrdinalType NumDofsPerNodeInInputArray, const Plato::OrdinalType NumDofsPerNodeInOutputArray>
void inline copy(const Plato::OrdinalType & aStride,
                 const Plato::OrdinalType & aNumVertices,
                 const Plato::ScalarVector & aInput,
                 Omega_h::Write<Omega_h::Real> & aOutput)
{
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, aNumVertices), LAMBDA_EXPRESSION(const Plato::OrdinalType & aIndex)
    {
        for(Plato::OrdinalType tIndex = 0; tIndex < NumDofsPerNodeInOutputArray; tIndex++)
        {
            Plato::OrdinalType tOutputDofIndex = (aIndex * NumDofsPerNodeInOutputArray) + tIndex;
            Plato::OrdinalType tInputDofIndex = (aIndex * NumDofsPerNodeInInputArray) + (aStride + tIndex);
            aOutput[tOutputDofIndex] = aInput(tInputDofIndex);
        }
    },"PlatoDriver::copy");
}

} // namespace Plato

#endif /* SRC_PLATO_PLATOUTILITIES_HPP_ */
