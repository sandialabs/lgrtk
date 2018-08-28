#include "LinearElasticMaterial.hpp"

namespace Plato {

/******************************************************************************/
template<>
::Plato::IsotropicLinearElasticMaterial<1>::
IsotropicLinearElasticMaterial(const Teuchos::ParameterList& paramList) :
LinearElasticMaterial<1>()
/******************************************************************************/
{
    m_poissonsRatio = paramList.get<double>("Poissons Ratio");
    m_youngsModulus = paramList.get<double>("Youngs Modulus");
    auto v = m_poissonsRatio;
    auto k = m_youngsModulus;
    auto c = k/((1.0+v)*(1.0-2.0*v));
    m_cellStiffness(0,0)=c*(1.0-v);
}
/******************************************************************************/
template<>
::Plato::IsotropicLinearElasticMaterial<2>::
IsotropicLinearElasticMaterial(const Teuchos::ParameterList& paramList) :
LinearElasticMaterial<2>()
/******************************************************************************/
{
    m_poissonsRatio = paramList.get<double>("Poissons Ratio");
    m_youngsModulus = paramList.get<double>("Youngs Modulus");
    auto v = m_poissonsRatio;
    auto k = m_youngsModulus;
    auto c = k/((1.0+v)*(1.0-2.0*v));
    m_cellStiffness(0,0)=c*(1.0-v); m_cellStiffness(0,1)=c*v;
    m_cellStiffness(1,0)=c*v;       m_cellStiffness(1,1)=c*(1.0-v);
    m_cellStiffness(2,2)=1.0/2.0*c*(1.0-2.0*v);
}
/******************************************************************************/
template<>
::Plato::IsotropicLinearElasticMaterial<3>::
IsotropicLinearElasticMaterial(const Teuchos::ParameterList& paramList) :
LinearElasticMaterial<3>()
/******************************************************************************/
{
    m_poissonsRatio = paramList.get<double>("Poissons Ratio");
    m_youngsModulus = paramList.get<double>("Youngs Modulus");
    auto v = m_poissonsRatio;
    auto k = m_youngsModulus;
    auto c = k/((1.0+v)*(1.0-2.0*v));
    m_cellStiffness(0,0)=c*(1.0-v); m_cellStiffness(0,1)=c*v;       m_cellStiffness(0,2)=c*v;
    m_cellStiffness(1,0)=c*v;       m_cellStiffness(1,1)=c*(1.0-v); m_cellStiffness(1,2)=c*v;
    m_cellStiffness(2,0)=c*v;       m_cellStiffness(2,1)=c*v;       m_cellStiffness(2,2)=c*(1.0-v);
    m_cellStiffness(3,3)=1.0/2.0*c*(1.0-2.0*v);
    m_cellStiffness(4,4)=1.0/2.0*c*(1.0-2.0*v);
    m_cellStiffness(5,5)=1.0/2.0*c*(1.0-2.0*v);
}

/******************************************************************************/
template<>
::Plato::IsotropicLinearElasticMaterial<1>::
IsotropicLinearElasticMaterial(const Plato::Scalar & aYoungsModulus, const Plato::Scalar & aPoissonsRatio) :
    LinearElasticMaterial<1>(),
    m_poissonsRatio(aPoissonsRatio),
    m_youngsModulus(aYoungsModulus)
/******************************************************************************/
{
    auto v = m_poissonsRatio;
    auto k = m_youngsModulus;
    auto c = k/((1.0+v)*(1.0-2.0*v));
    m_cellStiffness(0,0)=c*(1.0-v);
}

/******************************************************************************/
template<>
::Plato::IsotropicLinearElasticMaterial<2>::
IsotropicLinearElasticMaterial(const Plato::Scalar & aYoungsModulus, const Plato::Scalar & aPoissonsRatio) :
    LinearElasticMaterial<2>(),
    m_poissonsRatio(aPoissonsRatio),
    m_youngsModulus(aYoungsModulus)
/******************************************************************************/
{
    auto v = m_poissonsRatio;
    auto k = m_youngsModulus;
    auto c = k/((1.0+v)*(1.0-2.0*v));
    m_cellStiffness(0,0)=c*(1.0-v); m_cellStiffness(0,1)=c*v;
    m_cellStiffness(1,0)=c*v;       m_cellStiffness(1,1)=c*(1.0-v);
    m_cellStiffness(2,2)=1.0/2.0*c*(1.0-2.0*v);
}
/******************************************************************************/
template<>
::Plato::IsotropicLinearElasticMaterial<3>::
IsotropicLinearElasticMaterial(const Plato::Scalar & aYoungsModulus, const Plato::Scalar & aPoissonsRatio) :
    LinearElasticMaterial<3>(),
    m_poissonsRatio(aPoissonsRatio),
    m_youngsModulus(aYoungsModulus)
/******************************************************************************/
{
    auto v = m_poissonsRatio;
    auto k = m_youngsModulus;
    auto c = k/((1.0+v)*(1.0-2.0*v));
    m_cellStiffness(0,0)=c*(1.0-v); m_cellStiffness(0,1)=c*v;       m_cellStiffness(0,2)=c*v;
    m_cellStiffness(1,0)=c*v;       m_cellStiffness(1,1)=c*(1.0-v); m_cellStiffness(1,2)=c*v;
    m_cellStiffness(2,0)=c*v;       m_cellStiffness(2,1)=c*v;       m_cellStiffness(2,2)=c*(1.0-v);
    m_cellStiffness(3,3)=1.0/2.0*c*(1.0-2.0*v);
    m_cellStiffness(4,4)=1.0/2.0*c*(1.0-2.0*v);
    m_cellStiffness(5,5)=1.0/2.0*c*(1.0-2.0*v);
}

} // namespace Plato 
