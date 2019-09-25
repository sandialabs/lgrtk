#ifndef STRAIN_HPP
#define STRAIN_HPP

#include "plato/SimplexMechanics.hpp"
#include "plato/PlatoStaticsTypes.hpp"

#include <Omega_h_vector.hpp>

namespace Plato
{

/******************************************************************************/
/*! Strain functor.
  
    given a gradient matrix and displacement array, compute the strain.
    strain tensor in Voigt notation = {e_xx, e_yy, e_zz, e_yz, e_xz, e_xy}
*/
/******************************************************************************/
template<Plato::OrdinalType SpaceDim>
class Strain : public Plato::SimplexMechanics<SpaceDim>
{
  private:

    using Plato::SimplexMechanics<SpaceDim>::mNumVoigtTerms;
    using Plato::SimplexMechanics<SpaceDim>::mNumNodesPerCell;
    using Plato::SimplexMechanics<SpaceDim>::mNumDofsPerCell;

  public:

    template<typename ScalarType>
    DEVICE_TYPE inline void
    operator()( Plato::OrdinalType cellOrdinal,
                Kokkos::View<ScalarType**, Kokkos::LayoutRight, Plato::MemSpace> const& strain,
                Kokkos::View<ScalarType**, Kokkos::LayoutRight, Plato::MemSpace> const&  u,
                Omega_h::Vector<mNumVoigtTerms> const* gradientMatrix) const {

      // compute strain
      //
      for( Plato::OrdinalType iVoigt=0; iVoigt<mNumVoigtTerms; iVoigt++){
        strain(cellOrdinal,iVoigt) = 0.0;
        for( Plato::OrdinalType iDof=0; iDof<mNumDofsPerCell; iDof++){
          strain(cellOrdinal,iVoigt) += u(cellOrdinal,iDof)*gradientMatrix[iDof][iVoigt];
        }
      }
    }

    template<typename StrainScalarType, typename DispScalarType, typename GradientScalarType>
    DEVICE_TYPE inline void
    operator()( Plato::OrdinalType cellOrdinal,
                Plato::ScalarMultiVectorT< StrainScalarType   > const& strain,
                Plato::ScalarMultiVectorT< DispScalarType     > const& u,
                Plato::ScalarArray3DT<     GradientScalarType > const& gradient) const {

      Plato::OrdinalType voigtTerm=0;
      for(Plato::OrdinalType iDof=0; iDof<SpaceDim; iDof++){
        strain(cellOrdinal,voigtTerm)=0.0;
        for( Plato::OrdinalType iNode=0; iNode<mNumNodesPerCell; iNode++){
          Plato::OrdinalType localOrdinal = iNode*SpaceDim+iDof;
          strain(cellOrdinal,voigtTerm) += u(cellOrdinal,localOrdinal)*gradient(cellOrdinal,iNode,iDof);
        }
        voigtTerm++;
      }
      for (Plato::OrdinalType jDof=SpaceDim-1; jDof>=1; jDof--){
        for (Plato::OrdinalType iDof=jDof-1; iDof>=0; iDof--){
          for( Plato::OrdinalType iNode=0; iNode<mNumNodesPerCell; iNode++){
            Plato::OrdinalType iLocalOrdinal = iNode*SpaceDim+iDof;
            Plato::OrdinalType jLocalOrdinal = iNode*SpaceDim+jDof;
            strain(cellOrdinal,voigtTerm) +=(u(cellOrdinal,jLocalOrdinal)*gradient(cellOrdinal,iNode,iDof)
                                            +u(cellOrdinal,iLocalOrdinal)*gradient(cellOrdinal,iNode,jDof));
          }
          voigtTerm++;
        }
      }
    }
};

} // namespace Plato
#endif
