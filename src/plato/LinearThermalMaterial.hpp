#ifndef LINEARTHERMALMATERIAL_HPP
#define LINEARTHERMALMATERIAL_HPP

#include <Omega_h_matrix.hpp>
#include <Teuchos_ParameterList.hpp>

#include "plato/PlatoStaticsTypes.hpp"

namespace Plato {

/******************************************************************************/
/*!
  \brief Base class for Linear Thermal material models
*/
  template<int SpatialDim>
  class LinearThermalMaterial
/******************************************************************************/
{
  protected:
    Plato::Scalar mCellDensity;
    Plato::Scalar mCellSpecificHeat;
    Omega_h::Matrix<SpatialDim,SpatialDim> mCellConductivity;
  
  public:
    LinearThermalMaterial();
    Omega_h::Matrix<SpatialDim,SpatialDim> getConductivityMatrix() const {return mCellConductivity;}
    Plato::Scalar getMassDensity()  const {return mCellDensity;}
    Plato::Scalar getSpecificHeat() const {return mCellSpecificHeat;}
};

/******************************************************************************/
template<int SpatialDim>
LinearThermalMaterial<SpatialDim>::
LinearThermalMaterial() : mCellDensity(0.0), mCellSpecificHeat(0.0)
/******************************************************************************/
{
  for(int i=0; i<SpatialDim; i++)
    for(int j=0; j<SpatialDim; j++)
      mCellConductivity(i,j) = 0.0;
  
}

/******************************************************************************/
/*!
  \brief Derived class for isotropic linear elastic material model
*/
  template<int SpatialDim>
  class IsotropicLinearThermalMaterial : public LinearThermalMaterial<SpatialDim>
/******************************************************************************/
{
    using LinearThermalMaterial<SpatialDim>::mCellConductivity;
    using LinearThermalMaterial<SpatialDim>::mCellDensity;
    using LinearThermalMaterial<SpatialDim>::mCellSpecificHeat;
  public:
    IsotropicLinearThermalMaterial(const Teuchos::ParameterList& paramList) :
    LinearThermalMaterial<SpatialDim>()
    {
      auto t_conductivityCoef = paramList.get<Plato::Scalar>("Conductivity Coefficient");
      for(int i=0; i<SpatialDim; i++)
        mCellConductivity(i,i)=t_conductivityCoef;

      if( paramList.isType<Plato::Scalar>("Mass Density") ){
        mCellDensity = paramList.get<Plato::Scalar>("Mass Density");
      } else {
        mCellDensity = 1.0;
      }
      if( paramList.isType<Plato::Scalar>("Specific Heat") ){
        mCellSpecificHeat = paramList.get<Plato::Scalar>("Specific Heat");
      } else {
        mCellSpecificHeat = 1.0;
      }
    }
};




/******************************************************************************/
/*!
  \brief Factory for creating material models
*/
  template<int SpatialDim>
  class ThermalModelFactory
/******************************************************************************/
{
  public:
    ThermalModelFactory(const Teuchos::ParameterList& paramList) : mParamList(paramList) {}
    Teuchos::RCP<LinearThermalMaterial<SpatialDim>> create();
  private:
    const Teuchos::ParameterList& mParamList;
};
/******************************************************************************/
template<int SpatialDim>
Teuchos::RCP<LinearThermalMaterial<SpatialDim>>
ThermalModelFactory<SpatialDim>::create()
/******************************************************************************/
{
  auto modelParamList = mParamList.get<Teuchos::ParameterList>("Material Model");

  if( modelParamList.isSublist("Isotropic Linear Thermal") ){
    return Teuchos::rcp(new IsotropicLinearThermalMaterial<SpatialDim>(modelParamList.sublist("Isotropic Linear Thermal")));
  }
  return Teuchos::RCP<LinearThermalMaterial<SpatialDim>>(nullptr);
}

}

#endif
