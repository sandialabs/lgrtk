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
    Plato::Scalar k = paramList.get<double>("Conductivity Coefficient");
    Plato::Scalar a = paramList.get<double>("Thermal Expansion Coefficient");
    auto c = E/((1.0+v)*(1.0-2.0*v));
    m_cellStiffness(0,0)=c*(1.0-v);
    m_cellThermalConductivity(0,0)=k;
    m_cellThermalExpansion(0,0)=a;
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
    Plato::Scalar k = paramList.get<double>("Conductivity Coefficient");
    Plato::Scalar a = paramList.get<double>("Thermal Expansion Coefficient");
    auto c = E/((1.0+v)*(1.0-2.0*v));
    m_cellStiffness(0,0)=c*(1.0-v); m_cellStiffness(0,1)=c*v;
    m_cellStiffness(1,0)=c*v;       m_cellStiffness(1,1)=c*(1.0-v);
    m_cellStiffness(2,2)=1.0/2.0*c*(1.0-2.0*v);

    m_cellThermalConductivity(0,0)=k;
    m_cellThermalConductivity(1,1)=k;

    m_cellThermalExpansion(0,0)=a;
    m_cellThermalExpansion(0,1)=a;
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
    Plato::Scalar k = paramList.get<double>("Conductivity Coefficient");
    Plato::Scalar a = paramList.get<double>("Thermal Expansion Coefficient");
    auto c = E/((1.0+v)*(1.0-2.0*v));
    m_cellStiffness(0,0)=c*(1.0-v); m_cellStiffness(0,1)=c*v;       m_cellStiffness(0,2)=c*v;
    m_cellStiffness(1,0)=c*v;       m_cellStiffness(1,1)=c*(1.0-v); m_cellStiffness(1,2)=c*v;
    m_cellStiffness(2,0)=c*v;       m_cellStiffness(2,1)=c*v;       m_cellStiffness(2,2)=c*(1.0-v);
    m_cellStiffness(3,3)=1.0/2.0*c*(1.0-2.0*v);
    m_cellStiffness(4,4)=1.0/2.0*c*(1.0-2.0*v);
    m_cellStiffness(5,5)=1.0/2.0*c*(1.0-2.0*v);

    m_cellThermalConductivity(0,0)=k;
    m_cellThermalConductivity(1,1)=k;
    m_cellThermalConductivity(2,2)=k;

    m_cellThermalExpansion(0,0)=a;
    m_cellThermalExpansion(0,1)=a;
    m_cellThermalExpansion(0,2)=a;
}

} // namespace Plato 
