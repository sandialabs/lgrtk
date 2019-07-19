#ifndef KINEMATICS_HPP
#define KINEMATICS_HPP

#include "plato/SimplexMechanics.hpp"
#include "plato/PlatoStaticsTypes.hpp"

namespace Plato
{

/******************************************************************************/
/*! Two-field Mechanical kinematics functor.

    Given a gradient matrix and state array, compute the pressure gradient
    and symmetric gradient of the displacement.
*/
/******************************************************************************/
template<Plato::OrdinalType SpaceDim>
class StabilizedKinematics : public Plato::SimplexStabilizedMechanics<SpaceDim>
{
  private:

    using Plato::SimplexStabilizedMechanics<SpaceDim>::mNumVoigtTerms;
    using Plato::SimplexStabilizedMechanics<SpaceDim>::mNumNodesPerCell;
    using Plato::SimplexStabilizedMechanics<SpaceDim>::mNumDofsPerNode;
    using Plato::SimplexStabilizedMechanics<SpaceDim>::mPDofOffset;

  public:

    template<typename StrainScalarType, typename StateScalarType, typename GradientScalarType>
    DEVICE_TYPE inline void
    operator()( Plato::OrdinalType cellOrdinal,
                Plato::ScalarMultiVectorT< StrainScalarType   > const& strain,
                Plato::ScalarMultiVectorT< StrainScalarType   > const& pgrad,
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
    }
};

} // namespace Plato

#endif
