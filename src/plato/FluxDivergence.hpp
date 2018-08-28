#ifndef FLUX_DIVERGENCE
#define FLUX_DIVERGENCE

#include "plato/PlatoStaticsTypes.hpp"

/******************************************************************************/
/*! Flux divergence functor.
  
    Given a thermal flux, compute the flux divergence.
*/
/******************************************************************************/
template<int SpaceDim>
class FluxDivergence : public SimplexThermal<SpaceDim>
{
  private:

    using SimplexThermal<SpaceDim>::m_numNodesPerCell;
    using SimplexThermal<SpaceDim>::m_numDofsPerCell;

  public:

    template<
      typename ForcingScalarType, 
      typename FluxScalarType,
      typename GradientScalarType,
      typename VolumeScalarType>
    DEVICE_TYPE inline void
    operator()( int cellOrdinal,
                Plato::ScalarMultiVectorT< ForcingScalarType > q,
                Plato::ScalarMultiVectorT< FluxScalarType    > tflux,
                Plato::ScalarArray3DT<     GradientScalarType > gradient,
                Plato::ScalarVectorT<VolumeScalarType> cellVolume ) const {

      // compute flux divergence
      //
      for( int iNode=0; iNode<m_numNodesPerCell; iNode++){
        q(cellOrdinal,iNode) = 0.0;
        for(int iDim=0; iDim<SpaceDim; iDim++){
          q(cellOrdinal,iNode) += tflux(cellOrdinal,iDim)*gradient(cellOrdinal,iNode,iDim)*cellVolume(cellOrdinal);
        }
      }
    }
};
#endif
