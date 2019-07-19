#ifndef LGR_PLATO_LINEAR_STRESS_HPP
#define LGR_PLATO_LINEAR_STRESS_HPP

#include "plato/SimplexMechanics.hpp"

#include <Omega_h_matrix.hpp>

namespace Plato
{

/******************************************************************************/
/*! Stress functor.
  
    given a strain, compute the stress.
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

  public:

    LinearStress( const Omega_h::Matrix<mNumVoigtTerms,mNumVoigtTerms> cellStiffness) :
            mCellStiffness(cellStiffness) {}

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
          stress(cellOrdinal,iVoigt) += strain(cellOrdinal,jVoigt)*mCellStiffness(iVoigt, jVoigt);
        }
      }
    }
};
// class LinearStress

} // namespace Plato
#endif
