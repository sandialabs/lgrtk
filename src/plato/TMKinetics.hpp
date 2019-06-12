#ifndef LGR_PLATO_TMKINETICS_HPP
#define LGR_PLATO_TMKINETICS_HPP

#include "plato/SimplexThermomechanics.hpp"
#include "plato/LinearThermoelasticMaterial.hpp"

namespace Plato
{

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



    /***********************************************************************************
     * @brief Compute stress and thermal flux from strain, temperature, and temperature gradient
     * @param [in] aStrain infinitesimal strain tensor
     * @param [in] aTGrad temperature gradient
     * @param [in] aTemperature temperature
     * @param [out] aStress Cauchy stress tensor
     * @param [out] aFlux thermal flux vector
     **********************************************************************************/
    template<typename KineticsScalarType, typename KinematicsScalarType, typename StateScalarType>
    DEVICE_TYPE inline void
    operator()( int cellOrdinal,
                Kokkos::View<KineticsScalarType**,   Kokkos::LayoutRight, Plato::MemSpace> const& aStress,
                Kokkos::View<KineticsScalarType**,   Kokkos::LayoutRight, Plato::MemSpace> const& aFlux,
                Kokkos::View<KinematicsScalarType**, Kokkos::LayoutRight, Plato::MemSpace> const& aStrain,
                Kokkos::View<KinematicsScalarType**, Kokkos::LayoutRight, Plato::MemSpace> const& aTGrad,
                Kokkos::View<StateScalarType*,       Plato::MemSpace> const& aTemperature) const {

      // compute thermal strain
      //
      StateScalarType tstrain[m_numVoigtTerms] = {0};
      for( int iDim=0; iDim<SpaceDim; iDim++ ){
        tstrain[iDim] = m_cellThermalExpansionCoef * (aTemperature(cellOrdinal) - m_cellReferenceTemperature);
      }

      // compute stress
      //
      for( int iVoigt=0; iVoigt<m_numVoigtTerms; iVoigt++){
        aStress(cellOrdinal,iVoigt) = 0.0;
        for( int jVoigt=0; jVoigt<m_numVoigtTerms; jVoigt++){
          aStress(cellOrdinal,iVoigt) += (aStrain(cellOrdinal,jVoigt)-tstrain[jVoigt])*m_cellStiffness(iVoigt, jVoigt);
        }
      }

      // compute flux
      //
      for( int iDim=0; iDim<SpaceDim; iDim++){
        aFlux(cellOrdinal,iDim) = 0.0;
        for( int jDim=0; jDim<SpaceDim; jDim++){
          aFlux(cellOrdinal,iDim) += aTGrad(cellOrdinal,jDim)*m_cellThermalConductivity(iDim, jDim);
        }
      }
    }
};
// class TMKinetics


/******************************************************************************/
/*! Two-field thermoelastics functor.

    given: strain, pressure gradient, temperature gradient, fine scale
    displacement, pressure, and temperature

    compute: deviatoric stress, volume flux, cell stabilization, and thermal flux
*/
/******************************************************************************/
template<int SpaceDim>
class TwoFieldTMKinetics : public Plato::SimplexTwoFieldThermomechanics<SpaceDim>
{
  private:

    using Plato::SimplexTwoFieldThermomechanics<SpaceDim>::m_numVoigtTerms;
    using Plato::SimplexTwoFieldThermomechanics<SpaceDim>::m_numNodesPerCell;
    using Plato::SimplexTwoFieldThermomechanics<SpaceDim>::m_numDofsPerNode;
    using Plato::SimplexTwoFieldThermomechanics<SpaceDim>::m_numDofsPerCell;

    const Omega_h::Matrix<m_numVoigtTerms,m_numVoigtTerms> m_cellStiffness;
    const Plato::Scalar m_cellThermalExpansionCoef;
    const Omega_h::Matrix<SpaceDim, SpaceDim> m_cellThermalConductivity;
    const Plato::Scalar m_cellReferenceTemperature;
    Plato::Scalar m_BulkModulus, m_ShearModulus;

  public:

    TwoFieldTMKinetics( const Teuchos::RCP<Plato::LinearThermoelasticMaterial<SpaceDim>> materialModel ) :
            m_cellStiffness(materialModel->getStiffnessMatrix()),
            m_cellThermalExpansionCoef(materialModel->getThermalExpansion()),
            m_cellThermalConductivity(materialModel->getThermalConductivity()),
            m_cellReferenceTemperature(materialModel->getReferenceTemperature()),
            m_BulkModulus(0.0), m_ShearModulus(0.0)
    {
        for( int iDim=0; iDim<SpaceDim; iDim++ )
        {
            m_BulkModulus  += m_cellStiffness(iDim, iDim);
        }
        m_BulkModulus /= SpaceDim;

        int tNumShear = m_numVoigtTerms - SpaceDim;
        for( int iShear=0; iShear<tNumShear; iShear++ )
        {
            m_ShearModulus += m_cellStiffness(iShear+SpaceDim, iShear+SpaceDim);
        }
        m_ShearModulus /= tNumShear;
    }



    /***********************************************************************************
     * @brief Compute deviatoric stress, volume flux, cell stabilization, and thermal flux
     * @param [in] aStrain infinitesimal strain tensor
     * @param [in] aTGrad temperature gradient
     * @param [in] aTemperature temperature
     * @param [out] aStress Cauchy stress tensor
     * @param [out] aFlux thermal flux vector
     **********************************************************************************/
    template<typename KineticsScalarType, typename KinematicsScalarType, typename StateScalarType>
    DEVICE_TYPE inline void
    operator()( int cellOrdinal,
                Plato::ScalarVectorT      <KinematicsScalarType> const& aCellVolume,
                Plato::ScalarMultiVectorT <StateScalarType>      const& aProjectedPGrad,
                Plato::ScalarVectorT      <StateScalarType>      const& aPressure,
                Plato::ScalarVectorT      <StateScalarType>      const& aTemperature,
                Plato::ScalarMultiVectorT <KinematicsScalarType> const& aStrain,
                Plato::ScalarMultiVectorT <KinematicsScalarType> const& aPressureGrad,
                Plato::ScalarMultiVectorT <KinematicsScalarType> const& aTGrad,
                Plato::ScalarMultiVectorT <KineticsScalarType>   const& aDevStress,
                Plato::ScalarVectorT      <KineticsScalarType>   const& aVolumeFlux,
                Plato::ScalarMultiVectorT <KineticsScalarType>   const& aTFlux,
                Plato::ScalarMultiVectorT <KineticsScalarType>   const& aCellStabilization) const {

      // compute thermal strain and volume strain
      //
      StateScalarType tstrain[m_numVoigtTerms] = {0};
      StateScalarType tThermalVolStrain = 0.0;
      KinematicsScalarType tVolStrain = 0.0;
      for( int iDim=0; iDim<SpaceDim; iDim++ ){
        tstrain[iDim] = m_cellThermalExpansionCoef * (aTemperature(cellOrdinal) - m_cellReferenceTemperature);
        tThermalVolStrain += tstrain[iDim];
        tVolStrain += aStrain(cellOrdinal,iDim);
      }
      KinematicsScalarType tVolStrainVoigt[m_numVoigtTerms] = {0};
      StateScalarType tThermalVolStrainVoigt[m_numVoigtTerms] = {0};
      for( int iDim=0; iDim<SpaceDim; iDim++ ){
        tVolStrainVoigt[iDim] = tVolStrain;
        tThermalVolStrainVoigt[iDim] = tThermalVolStrain;
      }

      // compute deviatoric stress
      //
      for( int iVoigt=0; iVoigt<m_numVoigtTerms; iVoigt++){
        aDevStress(cellOrdinal,iVoigt) = 0.0;
        for( int jVoigt=0; jVoigt<m_numVoigtTerms; jVoigt++){
          aDevStress(cellOrdinal,iVoigt) += ( (aStrain(cellOrdinal,jVoigt)-tVolStrainVoigt[jVoigt])
                                             -(tstrain[jVoigt]-tThermalVolStrainVoigt[jVoigt]) ) *m_cellStiffness(iVoigt, jVoigt);
        }
      }

      // compute flux
      //
      for( int iDim=0; iDim<SpaceDim; iDim++){
        aTFlux(cellOrdinal,iDim) = 0.0;
        for( int jDim=0; jDim<SpaceDim; jDim++){
          aTFlux(cellOrdinal,iDim) += aTGrad(cellOrdinal,jDim)*m_cellThermalConductivity(iDim, jDim);
        }
      }

      // compute volume difference
      //
      aVolumeFlux(cellOrdinal) = tVolStrain - tThermalVolStrain - aPressure(cellOrdinal)/m_BulkModulus;

      // compute cell stabilization
      //
      KinematicsScalarType tTau = pow(aCellVolume(cellOrdinal),2.0/3.0)/(2.0*m_ShearModulus);
      for( int iDim=0; iDim<SpaceDim; iDim++){
          aCellStabilization(cellOrdinal,iDim) = tTau*(aPressureGrad(cellOrdinal,iDim) - aProjectedPGrad(cellOrdinal,iDim));
      }
    }
};

} // namespace Plato

#endif
