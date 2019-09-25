#ifndef LGR_PLATO_LINEAR_STRESS_HPP
#define LGR_PLATO_LINEAR_STRESS_HPP

#include "plato/SimplexMechanics.hpp"
#include "plato/LinearElasticMaterial.hpp"

#include <Omega_h_matrix.hpp>

namespace Plato
{

/******************************************************************************/
/*! Stress functor.
  
    given a strain, compute the stress.
    stress tensor in Voigt notation = {s_xx, s_yy, s_zz, s_yz, s_xz, s_xy}
*/
/******************************************************************************/
template<int SpaceDim>
class LinearStress : public Plato::SimplexMechanics<SpaceDim>
{
  private:

    using Plato::SimplexMechanics<SpaceDim>::mNumVoigtTerms;
    using Plato::SimplexMechanics<SpaceDim>::mNumNodesPerCell;
    using Plato::SimplexMechanics<SpaceDim>::mNumDofsPerNode;
    using Plato::SimplexMechanics<SpaceDim>::mNumDofsPerCell;

    const Omega_h::Matrix<mNumVoigtTerms,mNumVoigtTerms> mCellStiffness;
    Omega_h::Vector<mNumVoigtTerms> mReferenceStrain;

  public:


    LinearStress( const Omega_h::Matrix<mNumVoigtTerms,mNumVoigtTerms> aCellStiffness) :
            mCellStiffness(aCellStiffness) {
              for(int i=0; i<mNumVoigtTerms; i++)
                mReferenceStrain(i) = 0.0;
            }

    LinearStress(const Teuchos::RCP<Plato::LinearElasticMaterial<SpaceDim>> aMaterialModel ) :
            mCellStiffness(aMaterialModel->getStiffnessMatrix()),
            mReferenceStrain(aMaterialModel->getReferenceStrain()) {}

    template<typename StressScalarType, typename StrainScalarType>
    DEVICE_TYPE inline void
    operator()( int cellOrdinal,
                Kokkos::View<StressScalarType**, Kokkos::LayoutRight, Plato::MemSpace> const& stress,
                Kokkos::View<StrainScalarType**, Kokkos::LayoutRight, Plato::MemSpace> const& strain) const {

      // compute stress
      //
      for( int iVoigt=0; iVoigt<mNumVoigtTerms; iVoigt++){
        stress(cellOrdinal,iVoigt) = 0.0;
        for( int jVoigt=0; jVoigt<mNumVoigtTerms; jVoigt++){
          stress(cellOrdinal,iVoigt) += (strain(cellOrdinal,jVoigt)-mReferenceStrain(jVoigt))*mCellStiffness(iVoigt, jVoigt);
        }
      }
    }
};
// class LinearStress

} // namespace Plato
#endif
