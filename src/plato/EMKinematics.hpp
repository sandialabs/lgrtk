#ifndef EMKINEMATICS_HPP
#define EMKINEMATICS_HPP

#include "plato/SimplexElectromechanics.hpp"
#include "plato/PlatoStaticsTypes.hpp"

namespace Plato
{

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

    using Plato::SimplexElectromechanics<SpaceDim>::mNumVoigtTerms;
    using Plato::SimplexElectromechanics<SpaceDim>::mNumNodesPerCell;
    using Plato::SimplexElectromechanics<SpaceDim>::mNumDofsPerNode;

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
 
      // compute efield
      //
      Plato::OrdinalType dofOffset = SpaceDim;
      for(Plato::OrdinalType iDof=0; iDof<SpaceDim; iDof++){
        efield(cellOrdinal,iDof) = 0.0;
        for( Plato::OrdinalType iNode=0; iNode<mNumNodesPerCell; iNode++){
          Plato::OrdinalType localOrdinal = iNode*mNumDofsPerNode+dofOffset;
          efield(cellOrdinal,iDof) -= state(cellOrdinal,localOrdinal)*gradient(cellOrdinal,iNode,iDof);
        }
      }
    }
};

} // namespace Plato

#endif
