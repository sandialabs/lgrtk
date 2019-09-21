#include "plato/LinearThermoelasticMaterial.hpp"

namespace Plato {

/******************************************************************************/
template<>
::Plato::IsotropicLinearThermoelasticMaterial<1>::
IsotropicLinearThermoelasticMaterial(const Teuchos::ParameterList& paramList) :
LinearThermoelasticMaterial<1>(paramList)
/******************************************************************************/
{
    Plato::Scalar v = paramList.get<Plato::Scalar>("Poissons Ratio");
    Plato::Scalar E = paramList.get<Plato::Scalar>("Youngs Modulus");
    Plato::Scalar a = paramList.get<Plato::Scalar>("Thermal Expansion Coefficient");
    Plato::Scalar k = paramList.get<Plato::Scalar>("Thermal Conductivity Coefficient");
    auto c = E/((1.0+v)*(1.0-2.0*v));
    mCellStiffness(0,0)=c*(1.0-v);
    mCellThermalExpansionCoef(0)=a;
    mCellThermalConductivity(0,0)=k;
}
/******************************************************************************/
template<>
::Plato::IsotropicLinearThermoelasticMaterial<2>::
IsotropicLinearThermoelasticMaterial(const Teuchos::ParameterList& paramList) :
LinearThermoelasticMaterial<2>(paramList)
/******************************************************************************/
{
    Plato::Scalar v = paramList.get<Plato::Scalar>("Poissons Ratio");
    Plato::Scalar E = paramList.get<Plato::Scalar>("Youngs Modulus");
    Plato::Scalar a = paramList.get<Plato::Scalar>("Thermal Expansion Coefficient");
    Plato::Scalar k = paramList.get<Plato::Scalar>("Thermal Conductivity Coefficient");
    auto c = E/((1.0+v)*(1.0-2.0*v));
    mCellStiffness(0,0)=c*(1.0-v); mCellStiffness(0,1)=c*v;
    mCellStiffness(1,0)=c*v;       mCellStiffness(1,1)=c*(1.0-v);
    mCellStiffness(2,2)=1.0/2.0*c*(1.0-2.0*v);
    mCellThermalExpansionCoef(0)=a;
    mCellThermalExpansionCoef(1)=a;
    mCellThermalConductivity(0,0)=k;
    mCellThermalConductivity(1,1)=k;
}
/******************************************************************************/
template<>
::Plato::IsotropicLinearThermoelasticMaterial<3>::
IsotropicLinearThermoelasticMaterial(const Teuchos::ParameterList& paramList) :
LinearThermoelasticMaterial<3>(paramList)
/******************************************************************************/
{
    Plato::Scalar v = paramList.get<Plato::Scalar>("Poissons Ratio");
    Plato::Scalar E = paramList.get<Plato::Scalar>("Youngs Modulus");
    Plato::Scalar a = paramList.get<Plato::Scalar>("Thermal Expansion Coefficient");
    Plato::Scalar k = paramList.get<Plato::Scalar>("Thermal Conductivity Coefficient");
    auto c = E/((1.0+v)*(1.0-2.0*v));
    mCellStiffness(0,0)=c*(1.0-v); mCellStiffness(0,1)=c*v;       mCellStiffness(0,2)=c*v;
    mCellStiffness(1,0)=c*v;       mCellStiffness(1,1)=c*(1.0-v); mCellStiffness(1,2)=c*v;
    mCellStiffness(2,0)=c*v;       mCellStiffness(2,1)=c*v;       mCellStiffness(2,2)=c*(1.0-v);
    mCellStiffness(3,3)=1.0/2.0*c*(1.0-2.0*v);
    mCellStiffness(4,4)=1.0/2.0*c*(1.0-2.0*v);
    mCellStiffness(5,5)=1.0/2.0*c*(1.0-2.0*v);
    mCellThermalExpansionCoef(0)=a;
    mCellThermalExpansionCoef(1)=a;
    mCellThermalExpansionCoef(2)=a;
    mCellThermalConductivity(0,0)=k;
    mCellThermalConductivity(1,1)=k;
    mCellThermalConductivity(2,2)=k;
}

/******************************************************************************/
template<>
::Plato::CubicLinearThermoelasticMaterial<1>::
CubicLinearThermoelasticMaterial(const Teuchos::ParameterList& paramList) :
LinearThermoelasticMaterial<1>(paramList)
/******************************************************************************/
{
    Plato::Scalar v   = paramList.get<Plato::Scalar>("Poissons Ratio");
    Plato::Scalar E   = paramList.get<Plato::Scalar>("Youngs Modulus");
    Plato::Scalar a11 = paramList.get<Plato::Scalar>("a11");
    Plato::Scalar k11 = paramList.get<Plato::Scalar>("k11");
    auto c = E/((1.0+v)*(1.0-2.0*v));
    mCellStiffness(0,0)=c*(1.0-v);
    mCellThermalExpansionCoef(0)=a11;
    mCellThermalConductivity(0,0)=k11;
}
/******************************************************************************/
template<>
::Plato::CubicLinearThermoelasticMaterial<2>::
CubicLinearThermoelasticMaterial(const Teuchos::ParameterList& paramList) :
LinearThermoelasticMaterial<2>(paramList)
/******************************************************************************/
{
    Plato::Scalar v = paramList.get<Plato::Scalar>("Poissons Ratio");
    Plato::Scalar E = paramList.get<Plato::Scalar>("Youngs Modulus");
    Plato::Scalar a11 = paramList.get<Plato::Scalar>("a11");
    Plato::Scalar a22 = paramList.get<Plato::Scalar>("a22");
    Plato::Scalar k11 = paramList.get<Plato::Scalar>("k11");
    Plato::Scalar k22 = paramList.get<Plato::Scalar>("k22");
    auto c = E/((1.0+v)*(1.0-2.0*v));
    mCellStiffness(0,0)=c*(1.0-v); mCellStiffness(0,1)=c*v;
    mCellStiffness(1,0)=c*v;       mCellStiffness(1,1)=c*(1.0-v);
    mCellStiffness(2,2)=1.0/2.0*c*(1.0-2.0*v);
    mCellThermalExpansionCoef(0)=a11;
    mCellThermalExpansionCoef(1)=a22;
    mCellThermalConductivity(0,0)=k11;
    mCellThermalConductivity(1,1)=k22;
}
/******************************************************************************/
template<>
::Plato::CubicLinearThermoelasticMaterial<3>::
CubicLinearThermoelasticMaterial(const Teuchos::ParameterList& paramList) :
LinearThermoelasticMaterial<3>(paramList)
/******************************************************************************/
{
    Plato::Scalar v   = paramList.get<Plato::Scalar>("Poissons Ratio");
    Plato::Scalar E   = paramList.get<Plato::Scalar>("Youngs Modulus");
    Plato::Scalar a11 = paramList.get<Plato::Scalar>("a11");
    Plato::Scalar a22 = paramList.get<Plato::Scalar>("a22");
    Plato::Scalar a33 = paramList.get<Plato::Scalar>("a33");
    Plato::Scalar k11 = paramList.get<Plato::Scalar>("k11");
    Plato::Scalar k22 = paramList.get<Plato::Scalar>("k22");
    Plato::Scalar k33 = paramList.get<Plato::Scalar>("k33");
    auto c = E/((1.0+v)*(1.0-2.0*v));
    mCellStiffness(0,0)=c*(1.0-v); mCellStiffness(0,1)=c*v;       mCellStiffness(0,2)=c*v;
    mCellStiffness(1,0)=c*v;       mCellStiffness(1,1)=c*(1.0-v); mCellStiffness(1,2)=c*v;
    mCellStiffness(2,0)=c*v;       mCellStiffness(2,1)=c*v;       mCellStiffness(2,2)=c*(1.0-v);
    mCellStiffness(3,3)=1.0/2.0*c*(1.0-2.0*v);
    mCellStiffness(4,4)=1.0/2.0*c*(1.0-2.0*v);
    mCellStiffness(5,5)=1.0/2.0*c*(1.0-2.0*v);
    mCellThermalExpansionCoef(0)=a11;
    mCellThermalExpansionCoef(1)=a22;
    mCellThermalExpansionCoef(2)=a33;
    mCellThermalConductivity(0,0)=k11;
    mCellThermalConductivity(1,1)=k22;
    mCellThermalConductivity(2,2)=k33;
}

} // namespace Plato 
