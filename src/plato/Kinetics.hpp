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

    using Plato::SimplexStabilizedMechanics<SpaceDim>::m_numVoigtTerms;
    using Plato::SimplexStabilizedMechanics<SpaceDim>::m_numNodesPerCell;
    using Plato::SimplexStabilizedMechanics<SpaceDim>::m_numDofsPerNode;
    using Plato::SimplexStabilizedMechanics<SpaceDim>::m_numDofsPerCell;

    const Omega_h::Matrix<m_numVoigtTerms,m_numVoigtTerms> m_cellStiffness;
    Plato::Scalar m_BulkModulus, m_ShearModulus;

    const Plato::Scalar m_pressureScaling;

  public:

    StabilizedKinetics( const Teuchos::RCP<Plato::LinearElasticMaterial<SpaceDim>> materialModel ) :
            m_cellStiffness(materialModel->getStiffnessMatrix()),
            m_BulkModulus(0.0), m_ShearModulus(0.0),
            m_pressureScaling(materialModel->getPressureScaling())
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
      for( int iVoigt=0; iVoigt<m_numVoigtTerms; iVoigt++){
        aDevStress(cellOrdinal,iVoigt) = 0.0;
        for( int jVoigt=0; jVoigt<m_numVoigtTerms; jVoigt++){
          aDevStress(cellOrdinal,iVoigt) += aStrain(cellOrdinal,jVoigt) * m_cellStiffness(iVoigt, jVoigt);
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
      aPressure(cellOrdinal) *= m_pressureScaling;
      aVolumeFlux(cellOrdinal) = m_pressureScaling * (tVolStrain - aPressure(cellOrdinal)/m_BulkModulus);

      // compute cell stabilization
      //
      KinematicsScalarType tTau = pow(aCellVolume(cellOrdinal),2.0/3.0)/(2.0*m_ShearModulus);
      for( int iDim=0; iDim<SpaceDim; iDim++){
          aCellStabilization(cellOrdinal,iDim) = m_pressureScaling * tTau*(m_pressureScaling*aPressureGrad(cellOrdinal,iDim) - aProjectedPGrad(cellOrdinal,iDim));
      }
    }
};

} // namespace Plato

#endif
