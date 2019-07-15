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
      mAlpha = paramList.isType<double>("Alpha");
    }

    Plato::Scalar v = paramList.get<double>("Poissons Ratio");
    Plato::Scalar E = paramList.get<double>("Youngs Modulus");
    Plato::Scalar e33 = paramList.get<double>("e33");
    Plato::Scalar p33 = paramList.get<double>("p33");
    auto c = E/((1.0+v)*(1.0-2.0*v));
    mCellStiffness(0,0)=c*(1.0-v);
    mCellPiezoelectricCoupling(0,0)=e33;
    mCellPermittivity(0,0)=p33;
}
/******************************************************************************/
template<>
::Plato::IsotropicLinearElectroelasticMaterial<2>::
IsotropicLinearElectroelasticMaterial(const Teuchos::ParameterList& paramList) :
LinearElectroelasticMaterial<2>()
/******************************************************************************/
{
    if (paramList.isType<double>("Alpha")){
      mAlpha = paramList.isType<double>("Alpha");
    }

    Plato::Scalar v = paramList.get<double>("Poissons Ratio");
    Plato::Scalar E = paramList.get<double>("Youngs Modulus");
    Plato::Scalar e15 = paramList.get<double>("e15");
    Plato::Scalar e31 = paramList.get<double>("e31");
    Plato::Scalar e33 = paramList.get<double>("e33");
    Plato::Scalar p11 = paramList.get<double>("p11");
    Plato::Scalar p33 = paramList.get<double>("p33");
    auto c = E/((1.0+v)*(1.0-2.0*v));
    mCellStiffness(0,0)=c*(1.0-v); mCellStiffness(0,1)=c*v;
    mCellStiffness(1,0)=c*v;       mCellStiffness(1,1)=c*(1.0-v);
    mCellStiffness(2,2)=1.0/2.0*c*(1.0-2.0*v);

    mCellPiezoelectricCoupling(0,2)=e15;
    mCellPiezoelectricCoupling(1,0)=e31;
    mCellPiezoelectricCoupling(1,1)=e33;

    mCellPermittivity(0,0)=p11;
    mCellPermittivity(1,1)=p33;
}
/******************************************************************************/
template<>
::Plato::IsotropicLinearElectroelasticMaterial<3>::
IsotropicLinearElectroelasticMaterial(const Teuchos::ParameterList& paramList) :
LinearElectroelasticMaterial<3>()
/******************************************************************************/
{
    if (paramList.isType<double>("Alpha")){
      mAlpha = paramList.get<double>("Alpha");
    }

    Plato::Scalar v = paramList.get<double>("Poissons Ratio");
    Plato::Scalar E = paramList.get<double>("Youngs Modulus");
    Plato::Scalar e15 = paramList.get<double>("e15");
    Plato::Scalar e31 = paramList.get<double>("e31");
    Plato::Scalar e33 = paramList.get<double>("e33");
    Plato::Scalar p11 = paramList.get<double>("p11");
    Plato::Scalar p33 = paramList.get<double>("p33");
    auto c = E/((1.0+v)*(1.0-2.0*v));
    mCellStiffness(0,0)=c*(1.0-v); mCellStiffness(0,1)=c*v;       mCellStiffness(0,2)=c*v;
    mCellStiffness(1,0)=c*v;       mCellStiffness(1,1)=c*(1.0-v); mCellStiffness(1,2)=c*v;
    mCellStiffness(2,0)=c*v;       mCellStiffness(2,1)=c*v;       mCellStiffness(2,2)=c*(1.0-v);
    mCellStiffness(3,3)=1.0/2.0*c*(1.0-2.0*v);
    mCellStiffness(4,4)=1.0/2.0*c*(1.0-2.0*v);
    mCellStiffness(5,5)=1.0/2.0*c*(1.0-2.0*v);

    mCellPiezoelectricCoupling(2,0)=e31;
    mCellPiezoelectricCoupling(2,1)=e31;
    mCellPiezoelectricCoupling(2,2)=e33;
    mCellPiezoelectricCoupling(1,3)=e15;
    mCellPiezoelectricCoupling(0,4)=e15;

    mCellPermittivity(0,0)=p11;
    mCellPermittivity(1,1)=p11;
    mCellPermittivity(2,2)=p33;
}

} // namespace Plato 
