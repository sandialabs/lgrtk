#ifndef EMKINEMATICS_HPP
#define EMKINEMATICS_HPP

#include "plato/SimplexElectromechanics.hpp"
#include "plato/PlatoStaticsTypes.hpp"

/******************************************************************************/
/*! Electromechanical kinematics functor.
  
    Given a gradient matrix and displacement array, compute the strain 
    and electric field.
*/
/******************************************************************************/
template<Plato::OrdinalType SpaceDim>
class EMKinematics : public Plato::SimplexElectromechanics<SpaceDim>
{
  private:

    using Plato::SimplexElectromechanics<SpaceDim>::m_numVoigtTerms;
    using Plato::SimplexElectromechanics<SpaceDim>::m_numNodesPerCell;
    using Plato::SimplexElectromechanics<SpaceDim>::m_numDofsPerNode;

  public:

    template<typename StrainScalarType, typename StateScalarType, typename GradientScalarType>
    DEVICE_TYPE inline void
    operator()( Plato::OrdinalType cellOrdinal,
                Plato::ScalarMultiVectorT< StrainScalarType   > const& strain,
                Plato::ScalarMultiVectorT< StrainScalarType   > const& efield,
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
 
      // compute efield
      //
      Plato::OrdinalType dofOffset = SpaceDim-1;
      for(Plato::OrdinalType iDof=0; iDof<SpaceDim; iDof++){
        efield(cellOrdinal,iDof) = 0.0;
        for( Plato::OrdinalType iNode=0; iNode<m_numNodesPerCell; iNode++){
          Plato::OrdinalType localOrdinal = iNode*m_numDofsPerNode+dofOffset;
          efield(cellOrdinal,iDof) += state(cellOrdinal,localOrdinal)*gradient(cellOrdinal,iNode,iDof);
        }
      }
    }
};
#endif
