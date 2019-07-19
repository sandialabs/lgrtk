#ifndef TMKINEMATICS_HPP
#define TMKINEMATICS_HPP

#include "plato/SimplexThermomechanics.hpp"
#include "plato/PlatoStaticsTypes.hpp"

namespace Plato
{

/******************************************************************************/
/*! Thermomechanical kinematics functor.
  
    Given a gradient matrix and displacement array, compute the strain 
    and temperature gradient.
*/
/******************************************************************************/
template<Plato::OrdinalType SpaceDim>
class TMKinematics : public Plato::SimplexThermomechanics<SpaceDim>
{
  private:

    using Plato::SimplexThermomechanics<SpaceDim>::mNumVoigtTerms;
    using Plato::SimplexThermomechanics<SpaceDim>::mNumNodesPerCell;
    using Plato::SimplexThermomechanics<SpaceDim>::mNumDofsPerNode;

  public:

    template<typename StrainScalarType, typename StateScalarType, typename GradientScalarType>
    DEVICE_TYPE inline void
    operator()( Plato::OrdinalType cellOrdinal,
                Plato::ScalarMultiVectorT< StrainScalarType   > const& strain,
                Plato::ScalarMultiVectorT< StrainScalarType   > const& tgrad,
                Plato::ScalarMultiVectorT< StateScalarType    > const& state,
                Plato::ScalarArray3DT<     GradientScalarType > const& gradient) const {

      // compute strain
      //
      Plato::OrdinalType voigtTerm=0;
      for(Plato::OrdinalType iDof=0; iDof<SpaceDim; iDof++){
        strain(cellOrdinal,voigtTerm)=0.0;
        for( Plato::OrdinalType iNode=0; iNode<mNumNodesPerCell; iNode++){
          Plato::OrdinalType localOrdinal = iNode*mNumDofsPerNode+iDof;
          strain(cellOrdinal,voigtTerm) += state(cellOrdinal,localOrdinal)*gradient(cellOrdinal,iNode,iDof);
        }
        voigtTerm++;
      }
      for (Plato::OrdinalType jDof=SpaceDim-1; jDof>=1; jDof--){
        for (Plato::OrdinalType iDof=jDof-1; iDof>=0; iDof--){
          for( Plato::OrdinalType iNode=0; iNode<mNumNodesPerCell; iNode++){
            Plato::OrdinalType iLocalOrdinal = iNode*mNumDofsPerNode+iDof;
            Plato::OrdinalType jLocalOrdinal = iNode*mNumDofsPerNode+jDof;
            strain(cellOrdinal,voigtTerm) +=(state(cellOrdinal,jLocalOrdinal)*gradient(cellOrdinal,iNode,iDof)
                                            +state(cellOrdinal,iLocalOrdinal)*gradient(cellOrdinal,iNode,jDof));
          }
          voigtTerm++;
        }
      }
 
      // compute tgrad
      //
      Plato::OrdinalType dofOffset = SpaceDim;
      for(Plato::OrdinalType iDof=0; iDof<SpaceDim; iDof++){
        tgrad(cellOrdinal,iDof) = 0.0;
        for( Plato::OrdinalType iNode=0; iNode<mNumNodesPerCell; iNode++){
          Plato::OrdinalType localOrdinal = iNode*mNumDofsPerNode+dofOffset;
          tgrad(cellOrdinal,iDof) += state(cellOrdinal,localOrdinal)*gradient(cellOrdinal,iNode,iDof);
        }
      }
    }
};
// class TMKinematics

/******************************************************************************/
/*! Two-field thermomechanical kinematics functor.

    Given a gradient matrix and state array, compute the pressure gradient,
    temperature gradient, and symmetric gradient of the displacement.
*/
/******************************************************************************/
template<Plato::OrdinalType SpaceDim>
class StabilizedTMKinematics : public Plato::SimplexStabilizedThermomechanics<SpaceDim>
{
  private:

    using Plato::SimplexStabilizedThermomechanics<SpaceDim>::mNumVoigtTerms;
    using Plato::SimplexStabilizedThermomechanics<SpaceDim>::mNumNodesPerCell;
    using Plato::SimplexStabilizedThermomechanics<SpaceDim>::mNumDofsPerNode;
    using Plato::SimplexStabilizedThermomechanics<SpaceDim>::mPDofOffset;
    using Plato::SimplexStabilizedThermomechanics<SpaceDim>::mTDofOffset;

  public:

    template<typename StrainScalarType, typename StateScalarType, typename GradientScalarType>
    DEVICE_TYPE inline void
    operator()( Plato::OrdinalType cellOrdinal,
                Plato::ScalarMultiVectorT< StrainScalarType   > const& strain,
                Plato::ScalarMultiVectorT< StrainScalarType   > const& pgrad,
                Plato::ScalarMultiVectorT< StrainScalarType   > const& tgrad,
                Plato::ScalarMultiVectorT< StateScalarType    > const& state,
                Plato::ScalarArray3DT<     GradientScalarType > const& gradient) const {

      // compute strain
      //
      Plato::OrdinalType voigtTerm=0;
      for(Plato::OrdinalType iDof=0; iDof<SpaceDim; iDof++){
        strain(cellOrdinal,voigtTerm)=0.0;
        for( Plato::OrdinalType iNode=0; iNode<mNumNodesPerCell; iNode++){
          Plato::OrdinalType localOrdinal = iNode*mNumDofsPerNode+iDof;
          strain(cellOrdinal,voigtTerm) += state(cellOrdinal,localOrdinal)*gradient(cellOrdinal,iNode,iDof);
        }
        voigtTerm++;
      }
      for (Plato::OrdinalType jDof=SpaceDim-1; jDof>=1; jDof--){
        for (Plato::OrdinalType iDof=jDof-1; iDof>=0; iDof--){
          for( Plato::OrdinalType iNode=0; iNode<mNumNodesPerCell; iNode++){
            Plato::OrdinalType iLocalOrdinal = iNode*mNumDofsPerNode+iDof;
            Plato::OrdinalType jLocalOrdinal = iNode*mNumDofsPerNode+jDof;
            strain(cellOrdinal,voigtTerm) +=(state(cellOrdinal,jLocalOrdinal)*gradient(cellOrdinal,iNode,iDof)
                                            +state(cellOrdinal,iLocalOrdinal)*gradient(cellOrdinal,iNode,jDof));
          }
          voigtTerm++;
        }
      }

      // compute pgrad
      //
      for(Plato::OrdinalType iDof=0; iDof<SpaceDim; iDof++){
        pgrad(cellOrdinal,iDof) = 0.0;
        for( Plato::OrdinalType iNode=0; iNode<mNumNodesPerCell; iNode++){
          Plato::OrdinalType localOrdinal = iNode*mNumDofsPerNode+mPDofOffset;
          pgrad(cellOrdinal,iDof) += state(cellOrdinal,localOrdinal)*gradient(cellOrdinal,iNode,iDof);
        }
      }

      // compute tgrad
      //
      for(Plato::OrdinalType iDof=0; iDof<SpaceDim; iDof++){
        tgrad(cellOrdinal,iDof) = 0.0;
        for( Plato::OrdinalType iNode=0; iNode<mNumNodesPerCell; iNode++){
          Plato::OrdinalType localOrdinal = iNode*mNumDofsPerNode+mTDofOffset;
          tgrad(cellOrdinal,iDof) += state(cellOrdinal,localOrdinal)*gradient(cellOrdinal,iNode,iDof);
        }
      }
    }
};

} // namespace Plato

#endif
