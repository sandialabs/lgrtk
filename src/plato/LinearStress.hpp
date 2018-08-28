#ifndef LGR_PLATO_LINEAR_STRESS_HPP
#define LGR_PLATO_LINEAR_STRESS_HPP

#include "plato/SimplexMechanics.hpp"

/******************************************************************************/
/*! Stress functor.
  
    given a strain, compute the stress.
*/
/******************************************************************************/
template<int SpaceDim>
class LinearStress : public Plato::SimplexMechanics<SpaceDim>
{
  private:

    using Plato::SimplexMechanics<SpaceDim>::m_numVoigtTerms;
    using Plato::SimplexMechanics<SpaceDim>::m_numNodesPerCell;
    using Plato::SimplexMechanics<SpaceDim>::m_numDofsPerNode;
    using Plato::SimplexMechanics<SpaceDim>::m_numDofsPerCell;

    const Omega_h::Matrix<m_numVoigtTerms,m_numVoigtTerms> m_cellStiffness;

  public:

    LinearStress( const Omega_h::Matrix<m_numVoigtTerms,m_numVoigtTerms> cellStiffness) :
            m_cellStiffness(cellStiffness) {}

    template<typename StressScalarType, typename StrainScalarType>
    DEVICE_TYPE inline void
    operator()( int cellOrdinal,
                Kokkos::View<StressScalarType**, Kokkos::LayoutRight, Plato::MemSpace> const& stress,
                Kokkos::View<StrainScalarType**, Kokkos::LayoutRight, Plato::MemSpace> const& strain) const {

      // compute stress
      //
      for( int iVoigt=0; iVoigt<m_numVoigtTerms; iVoigt++){
        stress(cellOrdinal,iVoigt) = 0.0;
        for( int jVoigt=0; jVoigt<m_numVoigtTerms; jVoigt++){
          stress(cellOrdinal,iVoigt) += strain(cellOrdinal,jVoigt)*m_cellStiffness(iVoigt, jVoigt);
        }
      }
    }
};
#endif
