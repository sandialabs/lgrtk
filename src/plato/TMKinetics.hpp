#ifndef LGR_PLATO_TMKINETICS_HPP
#define LGR_PLATO_TMKINETICS_HPP

#include "plato/SimplexThermomechanics.hpp"
#include "plato/LinearThermoelasticMaterial.hpp"

/******************************************************************************/
/*! Thermoelastics functor.
  
    given a strain, temperature gradient, and temperature, compute the stress and flux
*/
/******************************************************************************/
template<int SpaceDim>
class TMKinetics : public Plato::SimplexThermomechanics<SpaceDim>
{
  private:

    using Plato::SimplexThermomechanics<SpaceDim>::m_numVoigtTerms;
    using Plato::SimplexThermomechanics<SpaceDim>::m_numNodesPerCell;
    using Plato::SimplexThermomechanics<SpaceDim>::m_numDofsPerNode;
    using Plato::SimplexThermomechanics<SpaceDim>::m_numDofsPerCell;

    const Omega_h::Matrix<m_numVoigtTerms,m_numVoigtTerms> m_cellStiffness;
    const Plato::Scalar m_cellThermalExpansionCoef;
    const Omega_h::Matrix<SpaceDim, SpaceDim> m_cellThermalConductivity;
    const Plato::Scalar m_cellReferenceTemperature;
 
  public:

    TMKinetics( const Teuchos::RCP<Plato::LinearThermoelasticMaterial<SpaceDim>> materialModel ) :
            m_cellStiffness(materialModel->getStiffnessMatrix()),
            m_cellThermalExpansionCoef(materialModel->getThermalExpansion()),
            m_cellThermalConductivity(materialModel->getThermalConductivity()),
            m_cellReferenceTemperature(materialModel->getReferenceTemperature()) {}

    template<typename KineticsScalarType, typename KinematicsScalarType, typename StateScalarType>
    DEVICE_TYPE inline void
    operator()( int cellOrdinal,
                Kokkos::View<KineticsScalarType**,   Kokkos::LayoutRight, Plato::MemSpace> const& stress,
                Kokkos::View<KineticsScalarType**,   Kokkos::LayoutRight, Plato::MemSpace> const& flux,
                Kokkos::View<KinematicsScalarType**, Kokkos::LayoutRight, Plato::MemSpace> const& strain,
                Kokkos::View<KinematicsScalarType**, Kokkos::LayoutRight, Plato::MemSpace> const& tgrad,
                Kokkos::View<StateScalarType*,       Plato::MemSpace> const& temperature) const {

      // compute thermal strain
      //
      StateScalarType tstrain[m_numVoigtTerms] = {0};
      for( int iDim=0; iDim<SpaceDim; iDim++ ){
        tstrain[iDim] = m_cellThermalExpansionCoef * (temperature(cellOrdinal) - m_cellReferenceTemperature);
      }

      // compute stress
      //
      for( int iVoigt=0; iVoigt<m_numVoigtTerms; iVoigt++){
        stress(cellOrdinal,iVoigt) = 0.0;
        for( int jVoigt=0; jVoigt<m_numVoigtTerms; jVoigt++){
          stress(cellOrdinal,iVoigt) += (strain(cellOrdinal,jVoigt)-tstrain[jVoigt])*m_cellStiffness(iVoigt, jVoigt);
        }
      }

      // compute flux
      //
      for( int iDim=0; iDim<SpaceDim; iDim++){
        flux(cellOrdinal,iDim) = 0.0;
        for( int jDim=0; jDim<SpaceDim; jDim++){
          flux(cellOrdinal,iDim) += tgrad(cellOrdinal,jDim)*m_cellThermalConductivity(iDim, jDim);
        }
      }
    }
};
#endif
