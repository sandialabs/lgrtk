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
 
  public:

    TMKinetics( const Teuchos::RCP<Plato::LinearThermoelasticMaterial<SpaceDim>> materialModel ) :
            mCellStiffness(materialModel->getStiffnessMatrix()),
            mCellThermalExpansionCoef(materialModel->getThermalExpansion()),
            mCellThermalConductivity(materialModel->getThermalConductivity()),
            mCellReferenceTemperature(materialModel->getReferenceTemperature()) {}

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
      StateScalarType tstrain[mNumVoigtTerms] = {0};
      for( int iDim=0; iDim<SpaceDim; iDim++ ){
        tstrain[iDim] = mCellThermalExpansionCoef * (temperature(cellOrdinal) - mCellReferenceTemperature);
      }

      // compute stress
      //
      for( int iVoigt=0; iVoigt<mNumVoigtTerms; iVoigt++){
        stress(cellOrdinal,iVoigt) = 0.0;
        for( int jVoigt=0; jVoigt<mNumVoigtTerms; jVoigt++){
          stress(cellOrdinal,iVoigt) += (strain(cellOrdinal,jVoigt)-tstrain[jVoigt])*mCellStiffness(iVoigt, jVoigt);
        }
      }

      // compute flux
      //
      for( int iDim=0; iDim<SpaceDim; iDim++){
        flux(cellOrdinal,iDim) = 0.0;
        for( int jDim=0; jDim<SpaceDim; jDim++){
          flux(cellOrdinal,iDim) += tgrad(cellOrdinal,jDim)*mCellThermalConductivity(iDim, jDim);
        }
      }
    }
};
// class TMKinetics

} // namespace Plato

#endif
