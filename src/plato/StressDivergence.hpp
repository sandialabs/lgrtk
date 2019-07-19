#ifndef STRESS_DIVERGENCE
#define STRESS_DIVERGENCE

#include "plato/SimplexMechanics.hpp"

namespace Plato
{

/******************************************************************************/
/*! Stress Divergence functor.
  
    Given a stress, compute the stress divergence.
*/
/******************************************************************************/
template<int SpaceDim, int NumDofsPerNode=SpaceDim, int DofOffset=0>
class StressDivergence : public Plato::SimplexMechanics<SpaceDim>
{
  private:

    using Plato::SimplexMechanics<SpaceDim>::mNumNodesPerCell;

    Plato::OrdinalType mVoigt[SpaceDim][SpaceDim];

  public:

    StressDivergence()
    {
      Plato::OrdinalType voigtTerm=0;
      for(Plato::OrdinalType iDof=0; iDof<SpaceDim; iDof++){
        mVoigt[iDof][iDof] = voigtTerm++;
      }
      for (Plato::OrdinalType jDof=SpaceDim-1; jDof>=1; jDof--){
        for (Plato::OrdinalType iDof=jDof-1; iDof>=0; iDof--){
          mVoigt[iDof][jDof] = voigtTerm;
          mVoigt[jDof][iDof] = voigtTerm++;
        }
      }
    }

    template<
      typename ForcingScalarType, 
      typename StressScalarType,
      typename GradientScalarType,
      typename VolumeScalarType>
    DEVICE_TYPE inline void
    operator()( Plato::OrdinalType cellOrdinal,
                Plato::ScalarMultiVectorT< ForcingScalarType  > forcing,
                Plato::ScalarMultiVectorT< StressScalarType   > stress,
                Plato::ScalarArray3DT<     GradientScalarType > gradient,
                Plato::ScalarVectorT<      VolumeScalarType   > cellVolume,
                Plato::Scalar scale = 1.0 ) const {

      for(Plato::OrdinalType iDim=0; iDim<SpaceDim; iDim++){
        for( Plato::OrdinalType iNode=0; iNode<mNumNodesPerCell; iNode++){
          Plato::OrdinalType localOrdinal = iNode*NumDofsPerNode+iDim+DofOffset;
          for(Plato::OrdinalType jDim=0; jDim<SpaceDim; jDim++){
            forcing(cellOrdinal,localOrdinal) += 
              scale*cellVolume(cellOrdinal)*stress(cellOrdinal,mVoigt[iDim][jDim])*gradient(cellOrdinal,iNode,jDim);
          }
        }
      }
    }
};

} // namespace Plato

#endif
