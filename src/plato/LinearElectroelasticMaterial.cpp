#include "plato/LinearElectroelasticMaterial.hpp"

namespace Plato {

/******************************************************************************/
template<>
::Plato::IsotropicLinearElectroelasticMaterial<1>::
IsotropicLinearElectroelasticMaterial(const Teuchos::ParameterList& paramList) :
LinearElectroelasticMaterial<1>()
/******************************************************************************/
{
    if (paramList.isType<double>("Alpha")){
      m_alpha = paramList.isType<double>("Alpha");
    }

    Plato::Scalar v = paramList.get<double>("Poissons Ratio");
    Plato::Scalar E = paramList.get<double>("Youngs Modulus");
    Plato::Scalar e33 = paramList.get<double>("e33");
    Plato::Scalar p33 = paramList.get<double>("p33");
    auto c = E/((1.0+v)*(1.0-2.0*v));
    m_cellStiffness(0,0)=c*(1.0-v);
    m_cellPiezoelectricCoupling(0,0)=e33;
    m_cellPermittivity(0,0)=p33;
}
/******************************************************************************/
template<>
::Plato::IsotropicLinearElectroelasticMaterial<2>::
IsotropicLinearElectroelasticMaterial(const Teuchos::ParameterList& paramList) :
LinearElectroelasticMaterial<2>()
/******************************************************************************/
{
    if (paramList.isType<double>("Alpha")){
      m_alpha = paramList.isType<double>("Alpha");
    }

    Plato::Scalar v = paramList.get<double>("Poissons Ratio");
    Plato::Scalar E = paramList.get<double>("Youngs Modulus");
    Plato::Scalar e15 = paramList.get<double>("e15");
    Plato::Scalar e31 = paramList.get<double>("e31");
    Plato::Scalar e33 = paramList.get<double>("e33");
    Plato::Scalar p11 = paramList.get<double>("p11");
    Plato::Scalar p33 = paramList.get<double>("p33");
    auto c = E/((1.0+v)*(1.0-2.0*v));
    m_cellStiffness(0,0)=c*(1.0-v); m_cellStiffness(0,1)=c*v;
    m_cellStiffness(1,0)=c*v;       m_cellStiffness(1,1)=c*(1.0-v);
    m_cellStiffness(2,2)=1.0/2.0*c*(1.0-2.0*v);

    m_cellPiezoelectricCoupling(0,2)=e15;
    m_cellPiezoelectricCoupling(1,0)=e31;
    m_cellPiezoelectricCoupling(1,1)=e33;

    m_cellPermittivity(0,0)=p11;
    m_cellPermittivity(1,1)=p33;
}
/******************************************************************************/
template<>
::Plato::IsotropicLinearElectroelasticMaterial<3>::
IsotropicLinearElectroelasticMaterial(const Teuchos::ParameterList& paramList) :
LinearElectroelasticMaterial<3>()
/******************************************************************************/
{
    if (paramList.isType<double>("Alpha")){
      m_alpha = paramList.isType<double>("Alpha");
    }

    Plato::Scalar v = paramList.get<double>("Poissons Ratio");
    Plato::Scalar E = paramList.get<double>("Youngs Modulus");
    Plato::Scalar e15 = paramList.get<double>("e15");
    Plato::Scalar e31 = paramList.get<double>("e31");
    Plato::Scalar e33 = paramList.get<double>("e33");
    Plato::Scalar p11 = paramList.get<double>("p11");
    Plato::Scalar p33 = paramList.get<double>("p33");
    auto c = E/((1.0+v)*(1.0-2.0*v));
    m_cellStiffness(0,0)=c*(1.0-v); m_cellStiffness(0,1)=c*v;       m_cellStiffness(0,2)=c*v;
    m_cellStiffness(1,0)=c*v;       m_cellStiffness(1,1)=c*(1.0-v); m_cellStiffness(1,2)=c*v;
    m_cellStiffness(2,0)=c*v;       m_cellStiffness(2,1)=c*v;       m_cellStiffness(2,2)=c*(1.0-v);
    m_cellStiffness(3,3)=1.0/2.0*c*(1.0-2.0*v);
    m_cellStiffness(4,4)=1.0/2.0*c*(1.0-2.0*v);
    m_cellStiffness(5,5)=1.0/2.0*c*(1.0-2.0*v);

    m_cellPiezoelectricCoupling(2,0)=e31;
    m_cellPiezoelectricCoupling(2,1)=e31;
    m_cellPiezoelectricCoupling(2,2)=e33;
    m_cellPiezoelectricCoupling(1,3)=e15;
    m_cellPiezoelectricCoupling(2,4)=e15;

    m_cellPermittivity(0,0)=p11;
    m_cellPermittivity(1,1)=p11;
    m_cellPermittivity(2,2)=p33;
}

} // namespace Plato 
