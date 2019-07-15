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
    mCellStiffness(0,0)=c*(1.0-v);
    mCellThermalExpansionCoef=a;
    mCellThermalConductivity(0,0)=k;
    mCellReferenceTemperature=t;

    if( paramList.isType<double>("Mass Density") ){
      mCellDensity = paramList.get<double>("Mass Density");
    } else {
      mCellDensity = 1.0;
    }
    if( paramList.isType<double>("Specific Heat") ){
      mCellSpecificHeat = paramList.get<double>("Specific Heat");
    } else {
      mCellSpecificHeat = 1.0;
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
    mCellStiffness(0,0)=c*(1.0-v); mCellStiffness(0,1)=c*v;
    mCellStiffness(1,0)=c*v;       mCellStiffness(1,1)=c*(1.0-v);
    mCellStiffness(2,2)=1.0/2.0*c*(1.0-2.0*v);
    mCellThermalExpansionCoef=a;
    mCellThermalConductivity(0,0)=k;
    mCellThermalConductivity(1,1)=k;
    mCellReferenceTemperature=t;

    if( paramList.isType<double>("Mass Density") ){
      mCellDensity = paramList.get<double>("Mass Density");
    } else {
      mCellDensity = 1.0;
    }
    if( paramList.isType<double>("Specific Heat") ){
      mCellSpecificHeat = paramList.get<double>("Specific Heat");
    } else {
      mCellSpecificHeat = 1.0;
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
    mCellStiffness(0,0)=c*(1.0-v); mCellStiffness(0,1)=c*v;       mCellStiffness(0,2)=c*v;
    mCellStiffness(1,0)=c*v;       mCellStiffness(1,1)=c*(1.0-v); mCellStiffness(1,2)=c*v;
    mCellStiffness(2,0)=c*v;       mCellStiffness(2,1)=c*v;       mCellStiffness(2,2)=c*(1.0-v);
    mCellStiffness(3,3)=1.0/2.0*c*(1.0-2.0*v);
    mCellStiffness(4,4)=1.0/2.0*c*(1.0-2.0*v);
    mCellStiffness(5,5)=1.0/2.0*c*(1.0-2.0*v);
    mCellThermalExpansionCoef=a;
    mCellThermalConductivity(0,0)=k;
    mCellThermalConductivity(1,1)=k;
    mCellThermalConductivity(2,2)=k;
    mCellReferenceTemperature=t;

    if( paramList.isType<double>("Mass Density") ){
      mCellDensity = paramList.get<double>("Mass Density");
    } else {
      mCellDensity = 1.0;
    }
    if( paramList.isType<double>("Specific Heat") ){
      mCellSpecificHeat = paramList.get<double>("Specific Heat");
    } else {
      mCellSpecificHeat = 1.0;
    }
}

} // namespace Plato 
