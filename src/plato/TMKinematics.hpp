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

    using Plato::SimplexThermomechanics<SpaceDim>::m_numVoigtTerms;
    using Plato::SimplexThermomechanics<SpaceDim>::m_numNodesPerCell;
    using Plato::SimplexThermomechanics<SpaceDim>::m_numDofsPerNode;

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
        for( Plato::OrdinalType iNode=0; iNode<m_numNodesPerCell; iNode++){
          Plato::OrdinalType localOrdinal = iNode*m_numDofsPerNode+iDof;
          strain(cellOrdinal,voigtTerm) += state(cellOrdinal,localOrdinal)*gradient(cellOrdinal,iNode,iDof);
        }
        voigtTerm++;
      }
      for (Plato::OrdinalType jDof=SpaceDim-1; jDof>=1; jDof--){
        for (Plato::OrdinalType iDof=jDof-1; iDof>=0; iDof--){
          for( Plato::OrdinalType iNode=0; iNode<m_numNodesPerCell; iNode++){
            Plato::OrdinalType iLocalOrdinal = iNode*m_numDofsPerNode+iDof;
            Plato::OrdinalType jLocalOrdinal = iNode*m_numDofsPerNode+jDof;
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
        for( Plato::OrdinalType iNode=0; iNode<m_numNodesPerCell; iNode++){
          Plato::OrdinalType localOrdinal = iNode*m_numDofsPerNode+dofOffset;
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
class TwoFieldTMKinematics : public Plato::SimplexTwoFieldThermomechanics<SpaceDim>
{
  private:

    using Plato::SimplexTwoFieldThermomechanics<SpaceDim>::m_numVoigtTerms;
    using Plato::SimplexTwoFieldThermomechanics<SpaceDim>::m_numNodesPerCell;
    using Plato::SimplexTwoFieldThermomechanics<SpaceDim>::m_numDofsPerNode;
    using Plato::SimplexTwoFieldThermomechanics<SpaceDim>::m_PDofOffset;
    using Plato::SimplexTwoFieldThermomechanics<SpaceDim>::m_TDofOffset;

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
        for( Plato::OrdinalType iNode=0; iNode<m_numNodesPerCell; iNode++){
          Plato::OrdinalType localOrdinal = iNode*m_numDofsPerNode+iDof;
          strain(cellOrdinal,voigtTerm) += state(cellOrdinal,localOrdinal)*gradient(cellOrdinal,iNode,iDof);
        }
        voigtTerm++;
      }
      for (Plato::OrdinalType jDof=SpaceDim-1; jDof>=1; jDof--){
        for (Plato::OrdinalType iDof=jDof-1; iDof>=0; iDof--){
          for( Plato::OrdinalType iNode=0; iNode<m_numNodesPerCell; iNode++){
            Plato::OrdinalType iLocalOrdinal = iNode*m_numDofsPerNode+iDof;
            Plato::OrdinalType jLocalOrdinal = iNode*m_numDofsPerNode+jDof;
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
        for( Plato::OrdinalType iNode=0; iNode<m_numNodesPerCell; iNode++){
          Plato::OrdinalType localOrdinal = iNode*m_numDofsPerNode+m_PDofOffset;
          pgrad(cellOrdinal,iDof) += state(cellOrdinal,localOrdinal)*gradient(cellOrdinal,iNode,iDof);
        }
      }

      // compute tgrad
      //
      for(Plato::OrdinalType iDof=0; iDof<SpaceDim; iDof++){
        tgrad(cellOrdinal,iDof) = 0.0;
        for( Plato::OrdinalType iNode=0; iNode<m_numNodesPerCell; iNode++){
          Plato::OrdinalType localOrdinal = iNode*m_numDofsPerNode+m_TDofOffset;
          tgrad(cellOrdinal,iDof) += state(cellOrdinal,localOrdinal)*gradient(cellOrdinal,iNode,iDof);
        }
      }
    }
};

} // namespace Plato

#endif
