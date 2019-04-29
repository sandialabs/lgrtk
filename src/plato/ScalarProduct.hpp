#ifndef SCALAR_PRODUCT_HPP
#define SCALAR_PRODUCT_HPP

#include <Omega_h_vector.hpp>

#include "plato/PlatoStaticsTypes.hpp"

namespace Plato
{

/******************************************************************************/
/*! Scalar product functor.
  
    Given two voigt tensors, compute the scalar product.
    Assumes single point integration.
*/
/******************************************************************************/
template<Plato::OrdinalType NumElements>
class ScalarProduct
{
  public:

    template<typename ProductScalarType, 
             typename Tensor1ScalarType, 
             typename Tensor2ScalarType,
             typename VolumeScalarType>
    DEVICE_TYPE inline void
    operator()( Plato::OrdinalType cellOrdinal,
                Plato::ScalarVectorT<ProductScalarType> scalarProduct,
                Plato::ScalarMultiVectorT<Tensor1ScalarType> voigtTensor1,
                Plato::ScalarMultiVectorT<Tensor2ScalarType> voigtTensor2,
                Plato::ScalarVectorT<VolumeScalarType> cellVolume,
                Plato::Scalar scale = 1.0 ) const {

      // compute scalar product
      //
      ProductScalarType inc(0.0);
      for( Plato::OrdinalType iVoigt=0; iVoigt<NumElements; iVoigt++){
        inc += voigtTensor1(cellOrdinal,iVoigt)*voigtTensor2(cellOrdinal,iVoigt);
      }
      scalarProduct(cellOrdinal) += scale*inc*cellVolume(cellOrdinal);
    }

    template<typename ProductScalarType, 
             typename Tensor1ScalarType, 
             typename VolumeScalarType>
    DEVICE_TYPE inline void
    operator()( Plato::OrdinalType cellOrdinal,
                Plato::ScalarVectorT<ProductScalarType> scalarProduct,
                Plato::ScalarMultiVectorT<Tensor1ScalarType> voigtTensor1,
                Omega_h::Vector<NumElements> voigtTensor2,
                Plato::ScalarVectorT<VolumeScalarType> cellVolume,
                Plato::Scalar scale = 1.0 ) const {

      // compute scalar product
      //
      ProductScalarType inc(0.0);
      for( Plato::OrdinalType iVoigt=0; iVoigt<NumElements; iVoigt++){
        inc += voigtTensor1(cellOrdinal,iVoigt)*voigtTensor2[iVoigt];
      }
      scalarProduct(cellOrdinal) += scale*inc*cellVolume(cellOrdinal);
    }
};

} // namespace Plato

#endif
