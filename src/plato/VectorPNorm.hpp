#ifndef VECTOR_P_NORM_HPP
#define VECTOR_P_NORM_HPP

#include "plato/PlatoStaticsTypes.hpp"

namespace Plato
{

/******************************************************************************/
/*! Vector p-norm functor.

 Given a vector, compute the p-norm.
 Assumes single point integration.
 */
/******************************************************************************/
template<Plato::OrdinalType VectorLength>
class VectorPNorm
{
public:

    template<typename ResultScalarType, typename VectorScalarType, typename VolumeScalarType>
    DEVICE_TYPE inline void operator()(Plato::OrdinalType aCellOrdinal,
                                       Plato::ScalarVectorT<ResultScalarType> aPnorm,
                                       Plato::ScalarMultiVectorT<VectorScalarType> aArgVector,
                                       Plato::OrdinalType aPvalue,
                                       Plato::ScalarVectorT<VolumeScalarType> aCellVolume) const
    {

        // compute scalar product
        //
        aPnorm(aCellOrdinal) = 0.0;
        for(Plato::OrdinalType iTerm = 0; iTerm < VectorLength; iTerm++)
        {
            aPnorm(aCellOrdinal) += aArgVector(aCellOrdinal, iTerm) * aArgVector(aCellOrdinal, iTerm);
        }
        aPnorm(aCellOrdinal) = pow(aPnorm(aCellOrdinal), aPvalue / 2.0);
        aPnorm(aCellOrdinal) *= aCellVolume(aCellOrdinal);
    }
};
// class VectorPNorm

}// namespace Plato

#endif
