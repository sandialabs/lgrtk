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
    Omega_h::Vector<m_numVoigtTerms> m_referenceStrain;

  public:


    LinearStress( const Omega_h::Matrix<m_numVoigtTerms,m_numVoigtTerms> aCellStiffness) :
            m_cellStiffness(aCellStiffness) {
              for(int i=0; i<m_numVoigtTerms; i++)
                m_referenceStrain(i) = 0.0;
            }

    LinearStress(const Teuchos::RCP<Plato::LinearElasticMaterial<SpaceDim>> aMaterialModel ) :
            m_cellStiffness(aMaterialModel->getStiffnessMatrix()),
            m_referenceStrain(aMaterialModel->getReferenceStrain()) {}

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
          stress(cellOrdinal,iVoigt) += (strain(cellOrdinal,jVoigt)-m_referenceStrain(jVoigt))*m_cellStiffness(iVoigt, jVoigt);
        }
      }
    }
};
// class LinearStress

} // namespace Plato
#endif
