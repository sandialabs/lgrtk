#ifndef LGR_PLATO_KINETICS_HPP
#define LGR_PLATO_KINETICS_HPP

#include "plato/SimplexMechanics.hpp"
#include "plato/LinearElasticMaterial.hpp"

namespace Plato
{

/******************************************************************************/
/*! Two-field Elasticity functor.

    given: strain, pressure gradient, fine scale displacement, pressure

    compute: deviatoric stress, volume flux, cell stabilization
*/
/******************************************************************************/
template<int SpaceDim>
class StabilizedKinetics : public Plato::SimplexStabilizedMechanics<SpaceDim>
{
  private:

    using Plato::SimplexStabilizedMechanics<SpaceDim>::mNumVoigtTerms;
    using Plato::SimplexStabilizedMechanics<SpaceDim>::mNumNodesPerCell;
    using Plato::SimplexStabilizedMechanics<SpaceDim>::mNumDofsPerNode;
    using Plato::SimplexStabilizedMechanics<SpaceDim>::mNumDofsPerCell;

    const Omega_h::Matrix<mNumVoigtTerms,mNumVoigtTerms> mCellStiffness;
    Plato::Scalar mBulkModulus, mShearModulus;

    const Plato::Scalar mPressureScaling;

  public:

    StabilizedKinetics( const Teuchos::RCP<Plato::LinearElasticMaterial<SpaceDim>> materialModel ) :
            mCellStiffness(materialModel->getStiffnessMatrix()),
            mBulkModulus(0.0), mShearModulus(0.0),
            mPressureScaling(materialModel->getPressureScaling())
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
     * @brief Compute deviatoric stress, volume flux, cell stabilization
     * @param [in] aStrain infinitesimal strain tensor
     * @param [out] aStress Cauchy stress tensor
     **********************************************************************************/
    template<
      typename KineticsScalarType,
      typename KinematicsScalarType,
      typename NodeStateScalarType,
      typename VolumeScalarType>
    DEVICE_TYPE inline void
    operator()( int cellOrdinal,
                Plato::ScalarVectorT      <VolumeScalarType>     const& aCellVolume,
                Plato::ScalarMultiVectorT <NodeStateScalarType>  const& aProjectedPGrad,
                Plato::ScalarVectorT      <KineticsScalarType>   const& aPressure,
                Plato::ScalarMultiVectorT <KinematicsScalarType> const& aStrain,
                Plato::ScalarMultiVectorT <KinematicsScalarType> const& aPressureGrad,
                Plato::ScalarMultiVectorT <KineticsScalarType>   const& aDevStress,
                Plato::ScalarVectorT      <KineticsScalarType>   const& aVolumeFlux,
                Plato::ScalarMultiVectorT <KineticsScalarType>   const& aCellStabilization) const {

      // compute thermal strain and volume strain
      //
      KinematicsScalarType tVolStrain = 0.0;
      for( int iDim=0; iDim<SpaceDim; iDim++ ){
        tVolStrain += aStrain(cellOrdinal,iDim);
      }

      // compute deviatoric stress
      //
      for( int iVoigt=0; iVoigt<mNumVoigtTerms; iVoigt++){
        aDevStress(cellOrdinal,iVoigt) = 0.0;
        for( int jVoigt=0; jVoigt<mNumVoigtTerms; jVoigt++){
          aDevStress(cellOrdinal,iVoigt) += aStrain(cellOrdinal,jVoigt) * mCellStiffness(iVoigt, jVoigt);
        }
      }
      KineticsScalarType trace(0.0);
      for( int iDim=0; iDim<SpaceDim; iDim++ ){
        trace += aDevStress(cellOrdinal,iDim);
      }
      for( int iDim=0; iDim<SpaceDim; iDim++ ){
        aDevStress(cellOrdinal,iDim) -= trace/3.0;
      }

      // compute volume difference
      //
      aPressure(cellOrdinal) *= mPressureScaling;
      aVolumeFlux(cellOrdinal) = mPressureScaling * (tVolStrain - aPressure(cellOrdinal)/mBulkModulus);

      // compute cell stabilization
      //
      KinematicsScalarType tTau = pow(aCellVolume(cellOrdinal),2.0/3.0)/(2.0*mShearModulus);
      for( int iDim=0; iDim<SpaceDim; iDim++){
          aCellStabilization(cellOrdinal,iDim) = mPressureScaling * tTau*(mPressureScaling*aPressureGrad(cellOrdinal,iDim) - aProjectedPGrad(cellOrdinal,iDim));
      }
    }
};

} // namespace Plato

#endif
