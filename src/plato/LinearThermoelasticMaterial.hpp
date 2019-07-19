#ifndef LINEARTHERMOELASTICMATERIAL_HPP
#define LINEARTHERMOELASTICMATERIAL_HPP

#include <Omega_h_matrix.hpp>
#include <Teuchos_ParameterList.hpp>

#include "plato/PlatoStaticsTypes.hpp"

namespace Plato {

/******************************************************************************/
/*!
  \brief Base class for Linear Thermoelastic material models
*/
  template<int SpatialDim>
  class LinearThermoelasticMaterial
/******************************************************************************/
{
  protected:
    static constexpr auto mNumVoigtTerms = (SpatialDim == 3) ? 6 : 
                                           ((SpatialDim == 2) ? 3 :
                                          (((SpatialDim == 1) ? 1 : 0)));
    static_assert(mNumVoigtTerms, "SpatialDim must be 1, 2, or 3.");

    Plato::Scalar mCellDensity;
    Plato::Scalar mCellSpecificHeat;
    Omega_h::Matrix<mNumVoigtTerms,mNumVoigtTerms> mCellStiffness;
    Plato::Scalar mCellThermalExpansionCoef;
    Omega_h::Matrix<SpatialDim, SpatialDim> mCellThermalConductivity;
    Plato::Scalar mCellReferenceTemperature;

    Plato::Scalar mTemperatureScaling;
    Plato::Scalar mPressureScaling;

  public:
    LinearThermoelasticMaterial();
    decltype(mCellDensity)               getMassDensity()          const {return mCellDensity;}
    decltype(mCellSpecificHeat)          getSpecificHeat()         const {return mCellSpecificHeat;}
    decltype(mCellStiffness)             getStiffnessMatrix()      const {return mCellStiffness;}
    decltype(mCellThermalExpansionCoef)  getThermalExpansion()     const {return mCellThermalExpansionCoef;}
    decltype(mCellThermalConductivity)   getThermalConductivity()  const {return mCellThermalConductivity;}
    decltype(mCellReferenceTemperature)  getReferenceTemperature() const {return mCellReferenceTemperature;}
    decltype(mTemperatureScaling)        getTemperatureScaling()   const {return mTemperatureScaling;}
    decltype(mPressureScaling)           getPressureScaling()      const {return mPressureScaling;}
};

/******************************************************************************/
template<int SpatialDim>
LinearThermoelasticMaterial<SpatialDim>::
LinearThermoelasticMaterial()
/******************************************************************************/
{
  for(int i=0; i<mNumVoigtTerms; i++)
    for(int j=0; j<mNumVoigtTerms; j++)
      mCellStiffness(i,j) = 0.0;

  mCellThermalExpansionCoef = 0.0;

  for(int i=0; i<SpatialDim; i++)
    for(int j=0; j<SpatialDim; j++)
      mCellThermalConductivity(i,j) = 0.0;

  mCellReferenceTemperature = 0.0;

  mTemperatureScaling = 1.0;
  mPressureScaling = 1.0;
}

/******************************************************************************/
/*!
  \brief Derived class for isotropic linear thermoelastic material model
*/
  template<int SpatialDim>
  class IsotropicLinearThermoelasticMaterial : public LinearThermoelasticMaterial<SpatialDim>
/******************************************************************************/
{
  public:
    IsotropicLinearThermoelasticMaterial(const Teuchos::ParameterList& paramList);
    virtual ~IsotropicLinearThermoelasticMaterial(){}
};
// class IsotropicLinearThermoelasticMaterial

/******************************************************************************/
/*!
  \brief Factory for creating material models
*/
  template<int SpatialDim>
  class ThermoelasticModelFactory
/******************************************************************************/
{
  public:
    ThermoelasticModelFactory(const Teuchos::ParameterList& paramList) : mParamList(paramList) {}
    Teuchos::RCP<Plato::LinearThermoelasticMaterial<SpatialDim>> create();
  private:
    const Teuchos::ParameterList& mParamList;
};
/******************************************************************************/
template<int SpatialDim>
Teuchos::RCP<LinearThermoelasticMaterial<SpatialDim>>
ThermoelasticModelFactory<SpatialDim>::create()
/******************************************************************************/
{
  auto modelParamList = mParamList.get<Teuchos::ParameterList>("Material Model");

  if( modelParamList.isSublist("Isotropic Linear Thermoelastic") ){
    return Teuchos::rcp(new Plato::IsotropicLinearThermoelasticMaterial<SpatialDim>(modelParamList.sublist("Isotropic Linear Thermoelastic")));
  }
  return Teuchos::RCP<Plato::LinearThermoelasticMaterial<SpatialDim>>(nullptr);
}

} // namespace Plato

#endif
