#ifndef VECTOR_P_NORM_HPP
#define VECTOR_P_NORM_HPP

#include "plato/PlatoStaticsTypes.hpp"

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

    template<typename ResultScalarType, 
             typename VectorScalarType, 
             typename VolumeScalarType>
    DEVICE_TYPE inline void
    operator()( Plato::OrdinalType cellOrdinal,
                Plato::ScalarVectorT<ResultScalarType> pnorm,
                Plato::ScalarMultiVectorT<VectorScalarType> argVector,
                Plato::OrdinalType p,
                Plato::ScalarVectorT<VolumeScalarType> cellVolume ) const {

      // compute scalar product
      //
      pnorm(cellOrdinal) = 0.0;
      for( Plato::OrdinalType iTerm=0; iTerm<VectorLength; iTerm++){
        pnorm(cellOrdinal) += argVector(cellOrdinal,iTerm)*argVector(cellOrdinal,iTerm);
      }
      pnorm(cellOrdinal) = pow(pnorm(cellOrdinal),p/2.0);
      pnorm(cellOrdinal) *= cellVolume(cellOrdinal);
    }
};
#endif
