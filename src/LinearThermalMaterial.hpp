#ifndef LINEARTHERMALMATERIAL_HPP
#define LINEARTHERMALMATERIAL_HPP

#include <Omega_h_matrix.hpp>
#include <Teuchos_ParameterList.hpp>

#include "StaticsTypes.hpp"

namespace lgr {

/******************************************************************************/
/*!
  \brief Base class for Linear Thermal material models
*/
  template<int SpatialDim>
  class LinearThermalMaterial
/******************************************************************************/
{
  protected:
    Omega_h::Matrix<SpatialDim,SpatialDim> m_cellConductivity;
  
  public:
    LinearThermalMaterial();
    Omega_h::Matrix<SpatialDim,SpatialDim> getConductivityMatrix() const {return m_cellConductivity;}
};

/******************************************************************************/
template<int SpatialDim>
LinearThermalMaterial<SpatialDim>::
LinearThermalMaterial()
/******************************************************************************/
{
  for(int i=0; i<SpatialDim; i++)
    for(int j=0; j<SpatialDim; j++)
      m_cellConductivity(i,j) = 0.0;
}

/******************************************************************************/
/*!
  \brief Derived class for isotropic linear elastic material model
*/
  template<int SpatialDim>
  class IsotropicLinearThermalMaterial : public LinearThermalMaterial<SpatialDim>
/******************************************************************************/
{
    using LinearThermalMaterial<SpatialDim>::m_cellConductivity;
  public:
    IsotropicLinearThermalMaterial(const Teuchos::ParameterList& paramList) :
    LinearThermalMaterial<SpatialDim>()
    {
      m_conductivityCoef = paramList.get<double>("Conductivity Coefficient");
      for(int i=0; i<SpatialDim; i++)
        m_cellConductivity(i,i)=m_conductivityCoef;
    }
  private:
    Scalar m_conductivityCoef;
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
    ThermalModelFactory(const Teuchos::ParameterList& paramList) : m_paramList(paramList) {}
    Teuchos::RCP<LinearThermalMaterial<SpatialDim>> create();
  private:
    const Teuchos::ParameterList& m_paramList;
};
/******************************************************************************/
template<int SpatialDim>
Teuchos::RCP<LinearThermalMaterial<SpatialDim>>
ThermalModelFactory<SpatialDim>::create()
/******************************************************************************/
{
  auto modelParamList = m_paramList.get<Teuchos::ParameterList>("Material Model");

  if( modelParamList.isSublist("Isotropic Linear Thermal") ){
    return Teuchos::rcp(new IsotropicLinearThermalMaterial<SpatialDim>(modelParamList.sublist("Isotropic Linear Thermal")));
  }
  return Teuchos::RCP<LinearThermalMaterial<SpatialDim>>(nullptr);
}

}

#endif
