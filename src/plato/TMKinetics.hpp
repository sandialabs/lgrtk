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

    using Plato::SimplexThermomechanics<SpaceDim>::mNumVoigtTerms;
    using Plato::SimplexThermomechanics<SpaceDim>::mNumNodesPerCell;
    using Plato::SimplexThermomechanics<SpaceDim>::mNumDofsPerNode;
    using Plato::SimplexThermomechanics<SpaceDim>::mNumDofsPerCell;

    const Omega_h::Matrix<mNumVoigtTerms,mNumVoigtTerms> mCellStiffness;
    const Plato::Scalar mCellThermalExpansionCoef;
    const Omega_h::Matrix<SpaceDim, SpaceDim> mCellThermalConductivity;
    const Plato::Scalar mCellReferenceTemperature;

    const Plato::Scalar mScaling;
    const Plato::Scalar mScaling2;
 
  public:

    TMKinetics( const Teuchos::RCP<Plato::LinearThermoelasticMaterial<SpaceDim>> materialModel ) :
            mCellStiffness(materialModel->getStiffnessMatrix()),
            mCellThermalExpansionCoef(materialModel->getThermalExpansion()),
            mCellThermalConductivity(materialModel->getThermalConductivity()),
            mCellReferenceTemperature(materialModel->getReferenceTemperature()),
            mScaling(materialModel->getTemperatureScaling()),
            mScaling2(mScaling*mScaling) {}


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
      StateScalarType tstrain[mNumVoigtTerms] = {0};
      for( int iDim=0; iDim<SpaceDim; iDim++ ){
        tstrain[iDim] = mScaling * mCellThermalExpansionCoef * (aTemperature(cellOrdinal) - mCellReferenceTemperature);
      }

      // compute stress
      //
      for( int iVoigt=0; iVoigt<mNumVoigtTerms; iVoigt++){
        aStress(cellOrdinal,iVoigt) = 0.0;
        for( int jVoigt=0; jVoigt<mNumVoigtTerms; jVoigt++){
          aStress(cellOrdinal,iVoigt) += (aStrain(cellOrdinal,jVoigt)-tstrain[jVoigt])*mCellStiffness(iVoigt, jVoigt);

        }
      }

      // compute flux
      //
      for( int iDim=0; iDim<SpaceDim; iDim++){
        aFlux(cellOrdinal,iDim) = 0.0;
        for( int jDim=0; jDim<SpaceDim; jDim++){
          aFlux(cellOrdinal,iDim) += mScaling2 * aTGrad(cellOrdinal,jDim)*mCellThermalConductivity(iDim, jDim);
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
class StabilizedTMKinetics : public Plato::SimplexStabilizedThermomechanics<SpaceDim>
{
  private:

    using Plato::SimplexStabilizedThermomechanics<SpaceDim>::mNumVoigtTerms;
    using Plato::SimplexStabilizedThermomechanics<SpaceDim>::mNumNodesPerCell;
    using Plato::SimplexStabilizedThermomechanics<SpaceDim>::mNumDofsPerNode;
    using Plato::SimplexStabilizedThermomechanics<SpaceDim>::mNumDofsPerCell;

    const Omega_h::Matrix<mNumVoigtTerms,mNumVoigtTerms> mCellStiffness;
    const Plato::Scalar mCellThermalExpansionCoef;
    const Omega_h::Matrix<SpaceDim, SpaceDim> mCellThermalConductivity;
    const Plato::Scalar mCellReferenceTemperature;
    Plato::Scalar mBulkModulus, mShearModulus;

    const Plato::Scalar mTemperatureScaling;
    const Plato::Scalar mTemperatureScaling2;

    const Plato::Scalar mPressureScaling;
    const Plato::Scalar mPressureScaling2;

  public:

    StabilizedTMKinetics( const Teuchos::RCP<Plato::LinearThermoelasticMaterial<SpaceDim>> materialModel ) :
            mCellStiffness(materialModel->getStiffnessMatrix()),
            mCellThermalExpansionCoef(materialModel->getThermalExpansion()),
            mCellThermalConductivity(materialModel->getThermalConductivity()),
            mCellReferenceTemperature(materialModel->getReferenceTemperature()),
            mBulkModulus(0.0), mShearModulus(0.0),
            mTemperatureScaling(materialModel->getTemperatureScaling()),
            mTemperatureScaling2(mTemperatureScaling*mTemperatureScaling),
            mPressureScaling(materialModel->getPressureScaling()),
            mPressureScaling2(mPressureScaling*mPressureScaling)
    {
        for( int iDim=0; iDim<SpaceDim; iDim++ )
        {
            mBulkModulus  += mCellStiffness(0, iDim);
        }
        mBulkModulus /= SpaceDim;

        int tNumShear = mNumVoigtTerms - SpaceDim;
        for( int iShear=0; iShear<tNumShear; iShear++ )
        {
            mShearModulus += mCellStiffness(iShear+SpaceDim, iShear+SpaceDim);
        }
        mShearModulus /= tNumShear;
    }



    /***********************************************************************************
     * @brief Compute deviatoric stress, volume flux, cell stabilization, and thermal flux
     * @param [in] aStrain infinitesimal strain tensor
     * @param [in] aTGrad temperature gradient
     * @param [in] aTemperature temperature
     * @param [out] aStress Cauchy stress tensor
     * @param [out] aFlux thermal flux vector
     **********************************************************************************/
    template<
      typename KineticsScalarType,
      typename KinematicsScalarType,
      typename StateScalarType,
      typename NodeStateScalarType,
      typename VolumeScalarType>
    DEVICE_TYPE inline void
    operator()( int cellOrdinal,
                Plato::ScalarVectorT      <VolumeScalarType>     const& aCellVolume,
                Plato::ScalarMultiVectorT <NodeStateScalarType>  const& aProjectedPGrad,
                Plato::ScalarVectorT      <KineticsScalarType>   const& aPressure,
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
      StateScalarType tstrain[mNumVoigtTerms] = {0};
      StateScalarType tThermalVolStrain = 0.0;
      KinematicsScalarType tVolStrain = 0.0;
      for( int iDim=0; iDim<SpaceDim; iDim++ ){
        tstrain[iDim] = mTemperatureScaling * mCellThermalExpansionCoef * (aTemperature(cellOrdinal) - mCellReferenceTemperature);
        tThermalVolStrain += tstrain[iDim];
        tVolStrain += aStrain(cellOrdinal,iDim);
      }

      // compute deviatoric stress
      //
      for( int iVoigt=0; iVoigt<mNumVoigtTerms; iVoigt++){
        aDevStress(cellOrdinal,iVoigt) = 0.0;
        for( int jVoigt=0; jVoigt<mNumVoigtTerms; jVoigt++){
          aDevStress(cellOrdinal,iVoigt) += ( (aStrain(cellOrdinal,jVoigt)-tstrain[jVoigt]) ) *mCellStiffness(iVoigt, jVoigt);
        }
      }
      KineticsScalarType trace(0.0);
      for( int iDim=0; iDim<SpaceDim; iDim++ ){
        trace += aDevStress(cellOrdinal,iDim);
      }
      for( int iDim=0; iDim<SpaceDim; iDim++ ){
        aDevStress(cellOrdinal,iDim) -= trace/3.0;
      }

      // compute flux
      //
      for( int iDim=0; iDim<SpaceDim; iDim++){
        aTFlux(cellOrdinal,iDim) = 0.0;
        for( int jDim=0; jDim<SpaceDim; jDim++){
          aTFlux(cellOrdinal,iDim) += mTemperatureScaling2 * aTGrad(cellOrdinal,jDim)*mCellThermalConductivity(iDim, jDim);
        }
      }

      // compute volume difference
      //
      aPressure(cellOrdinal) *= mPressureScaling;
      aVolumeFlux(cellOrdinal) = mPressureScaling * (tVolStrain - tThermalVolStrain - aPressure(cellOrdinal)/mBulkModulus);

      // compute cell stabilization
      //
      KinematicsScalarType tTau = pow(aCellVolume(cellOrdinal),2.0/3.0)/(2.0*mShearModulus);
      for( int iDim=0; iDim<SpaceDim; iDim++){
          aCellStabilization(cellOrdinal,iDim) = mPressureScaling * tTau *
            (mPressureScaling*aPressureGrad(cellOrdinal,iDim) - aProjectedPGrad(cellOrdinal,iDim));
      }
    }
};

} // namespace Plato

#endif
