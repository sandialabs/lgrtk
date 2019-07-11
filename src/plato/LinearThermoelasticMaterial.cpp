#include "plato/LinearThermoelasticMaterial.hpp"

namespace Plato {

/******************************************************************************/
template<>
::Plato::IsotropicLinearThermoelasticMaterial<1>::
IsotropicLinearThermoelasticMaterial(const Teuchos::ParameterList& paramList) :
LinearThermoelasticMaterial<1>()
/******************************************************************************/
{
    Plato::Scalar v = paramList.get<double>("Poissons Ratio");
    Plato::Scalar E = paramList.get<double>("Youngs Modulus");
    Plato::Scalar a = paramList.get<double>("Thermal Expansion Coefficient");
    Plato::Scalar k = paramList.get<double>("Thermal Conductivity Coefficient");
    Plato::Scalar t = paramList.get<double>("Reference Temperature");
    auto c = E/((1.0+v)*(1.0-2.0*v));
    m_cellStiffness(0,0)=c*(1.0-v);
    m_cellThermalExpansionCoef=a;
    m_cellThermalConductivity(0,0)=k;
    m_cellReferenceTemperature=t;

    if( paramList.isType<double>("Mass Density") ){
      m_cellDensity = paramList.get<double>("Mass Density");
    } else {
      m_cellDensity = 1.0;
    }
    if( paramList.isType<double>("Specific Heat") ){
      m_cellSpecificHeat = paramList.get<double>("Specific Heat");
    } else {
      m_cellSpecificHeat = 1.0;
    }
    if( paramList.isType<double>("Temperature Scaling") ){
      m_temperatureScaling = paramList.get<double>("Temperature Scaling");
    } else {
      m_temperatureScaling = 1.0;
    }
    if( paramList.isType<double>("Pressure Scaling") ){
      m_pressureScaling = paramList.get<double>("Pressure Scaling");
    } else {
      m_pressureScaling = 1.0;
    }
}
/******************************************************************************/
template<>
::Plato::IsotropicLinearThermoelasticMaterial<2>::
IsotropicLinearThermoelasticMaterial(const Teuchos::ParameterList& paramList) :
LinearThermoelasticMaterial<2>()
/******************************************************************************/
{
    Plato::Scalar v = paramList.get<double>("Poissons Ratio");
    Plato::Scalar E = paramList.get<double>("Youngs Modulus");
    Plato::Scalar a = paramList.get<double>("Thermal Expansion Coefficient");
    Plato::Scalar k = paramList.get<double>("Thermal Conductivity Coefficient");
    Plato::Scalar t = paramList.get<double>("Reference Temperature");
    auto c = E/((1.0+v)*(1.0-2.0*v));
    m_cellStiffness(0,0)=c*(1.0-v); m_cellStiffness(0,1)=c*v;
    m_cellStiffness(1,0)=c*v;       m_cellStiffness(1,1)=c*(1.0-v);
    m_cellStiffness(2,2)=1.0/2.0*c*(1.0-2.0*v);
    m_cellThermalExpansionCoef=a;
    m_cellThermalConductivity(0,0)=k;
    m_cellThermalConductivity(1,1)=k;
    m_cellReferenceTemperature=t;

    if( paramList.isType<double>("Mass Density") ){
      m_cellDensity = paramList.get<double>("Mass Density");
    } else {
      m_cellDensity = 1.0;
    }
    if( paramList.isType<double>("Specific Heat") ){
      m_cellSpecificHeat = paramList.get<double>("Specific Heat");
    } else {
      m_cellSpecificHeat = 1.0;
    }
    if( paramList.isType<double>("Temperature Scaling") ){
      m_temperatureScaling = paramList.get<double>("Temperature Scaling");
    } else {
      m_temperatureScaling = 1.0;
    }
    if( paramList.isType<double>("Pressure Scaling") ){
      m_pressureScaling = paramList.get<double>("Pressure Scaling");
    } else {
      m_pressureScaling = 1.0;
    }
}
/******************************************************************************/
template<>
::Plato::IsotropicLinearThermoelasticMaterial<3>::
IsotropicLinearThermoelasticMaterial(const Teuchos::ParameterList& paramList) :
LinearThermoelasticMaterial<3>()
/******************************************************************************/
{
    Plato::Scalar v = paramList.get<double>("Poissons Ratio");
    Plato::Scalar E = paramList.get<double>("Youngs Modulus");
    Plato::Scalar a = paramList.get<double>("Thermal Expansion Coefficient");
    Plato::Scalar k = paramList.get<double>("Thermal Conductivity Coefficient");
    Plato::Scalar t = paramList.get<double>("Reference Temperature");
    auto c = E/((1.0+v)*(1.0-2.0*v));
    m_cellStiffness(0,0)=c*(1.0-v); m_cellStiffness(0,1)=c*v;       m_cellStiffness(0,2)=c*v;
    m_cellStiffness(1,0)=c*v;       m_cellStiffness(1,1)=c*(1.0-v); m_cellStiffness(1,2)=c*v;
    m_cellStiffness(2,0)=c*v;       m_cellStiffness(2,1)=c*v;       m_cellStiffness(2,2)=c*(1.0-v);
    m_cellStiffness(3,3)=1.0/2.0*c*(1.0-2.0*v);
    m_cellStiffness(4,4)=1.0/2.0*c*(1.0-2.0*v);
    m_cellStiffness(5,5)=1.0/2.0*c*(1.0-2.0*v);
    m_cellThermalExpansionCoef=a;
    m_cellThermalConductivity(0,0)=k;
    m_cellThermalConductivity(1,1)=k;
    m_cellThermalConductivity(2,2)=k;
    m_cellReferenceTemperature=t;

    if( paramList.isType<double>("Mass Density") ){
      m_cellDensity = paramList.get<double>("Mass Density");
    } else {
      m_cellDensity = 1.0;
    }
    if( paramList.isType<double>("Specific Heat") ){
      m_cellSpecificHeat = paramList.get<double>("Specific Heat");
    } else {
      m_cellSpecificHeat = 1.0;
    }
    if( paramList.isType<double>("Temperature Scaling") ){
      m_temperatureScaling = paramList.get<double>("Temperature Scaling");
    } else {
      m_temperatureScaling = 1.0;
    }
    if( paramList.isType<double>("Pressure Scaling") ){
      m_pressureScaling = paramList.get<double>("Pressure Scaling");
    } else {
      m_pressureScaling = 1.0;
    }
}

} // namespace Plato 
