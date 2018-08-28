#ifndef LGR_PLATO_HOMOGENIZED_STRESS_HPP
#define LGR_PLATO_HOMOGENIZED_STRESS_HPP

#include "plato/SimplexMechanics.hpp"

/******************************************************************************/
/*! Homogenized stress functor.
  
    given a characteristic strain, compute the homogenized stress.
*/
/******************************************************************************/
template<int SpaceDim>
class HomogenizedStress : public Plato::SimplexMechanics<SpaceDim>
{
  private:

    using Plato::SimplexMechanics<SpaceDim>::m_numVoigtTerms;
    using Plato::SimplexMechanics<SpaceDim>::m_numNodesPerCell;
    using Plato::SimplexMechanics<SpaceDim>::m_numDofsPerNode;
    using Plato::SimplexMechanics<SpaceDim>::m_numDofsPerCell;

    const Omega_h::Matrix<m_numVoigtTerms,m_numVoigtTerms> m_cellStiffness;
    const int m_columnIndex;

  public:

    HomogenizedStress( const Omega_h::Matrix<m_numVoigtTerms,m_numVoigtTerms> aCellStiffness, int aColumnIndex) :
            m_cellStiffness(aCellStiffness), 
            m_columnIndex(aColumnIndex) {}

    template<typename StressScalarType, typename StrainScalarType>
    DEVICE_TYPE inline void
    operator()( int cellOrdinal,
                Kokkos::View<StressScalarType**, Kokkos::LayoutRight, Plato::MemSpace> const& stress,
                Kokkos::View<StrainScalarType**, Kokkos::LayoutRight, Plato::MemSpace> const& strain) const {

      // compute stress
      //
      for( int iVoigt=0; iVoigt<m_numVoigtTerms; iVoigt++){
        stress(cellOrdinal,iVoigt) = m_cellStiffness(m_columnIndex, iVoigt);
        for( int jVoigt=0; jVoigt<m_numVoigtTerms; jVoigt++){
          stress(cellOrdinal,iVoigt) -= strain(cellOrdinal,jVoigt)*m_cellStiffness(m_columnIndex, jVoigt);
        }
      }
    }
};
#endif
