#ifndef PRESSURE_DIVERGENCE
#define PRESSURE_DIVERGENCE

#include "plato/SimplexMechanics.hpp"

namespace Plato
{

/******************************************************************************/
/*! Pressure Divergence functor.
  
    Given a pressure, p, and divergence matrix, b, compute the pressure divergence.

    f_{e,j(I,i)} = s v p_{e} b_{e,I,i}

            e:  element index
            I:  local node index
            i:  dimension index
            s:  scale factor.  1.0 by default.
            v:  cell volume.  Single value per cell, so single point integration is assumed.
        p_{e}:  pressure in element, e
    b_{e,I,i}:  basis derivative of local node I of element e with respect to dimenion i
       j(I,i):  strided 1D index of local dof i for local node I.
                j(I,i) = I*NumDofsPerNode + i + DofOffset

*/
/******************************************************************************/
template<int SpaceDim, int NumDofsPerNode=SpaceDim, int DofOffset=0>
class PressureDivergence : public Plato::SimplexMechanics<SpaceDim>
{
  private:

    using Plato::SimplexMechanics<SpaceDim>::m_numNodesPerCell;

  public:
    template<
      typename ForcingScalarType, 
      typename PressureScalarType,
      typename GradientScalarType,
      typename VolumeScalarType>
    DEVICE_TYPE inline void
    operator()( Plato::OrdinalType cellOrdinal,
                Plato::ScalarMultiVectorT <ForcingScalarType>  forcing,
                Plato::ScalarVectorT      <PressureScalarType> pressure,
                Plato::ScalarArray3DT     <GradientScalarType> gradient,
                Plato::ScalarVectorT      <VolumeScalarType>   cellVolume,
                Plato::Scalar scale = 1.0 ) const {

      for(Plato::OrdinalType iDim=0; iDim<SpaceDim; iDim++){
        for( Plato::OrdinalType iNode=0; iNode<m_numNodesPerCell; iNode++){
          Plato::OrdinalType localOrdinal = iNode*NumDofsPerNode+iDim+DofOffset;
          forcing(cellOrdinal,localOrdinal) += 
            scale*cellVolume(cellOrdinal)*pressure(cellOrdinal)*gradient(cellOrdinal,iNode,iDim);
        }
      }
    }
};

} // namespace Plato

#endif
