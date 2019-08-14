#include "plato/LinearElasticMaterial.hpp"

namespace Plato {

/******************************************************************************/
template<>
::Plato::IsotropicLinearElasticMaterial<1>::
IsotropicLinearElasticMaterial(const Teuchos::ParameterList& paramList) :
LinearElasticMaterial<1>(paramList)
/******************************************************************************/
{
    mPoissonsRatio = paramList.get<double>("Poissons Ratio");
    mYoungsModulus = paramList.get<double>("Youngs Modulus");
    auto v = mPoissonsRatio;
    auto k = mYoungsModulus;
    auto c = k/((1.0+v)*(1.0-2.0*v));
    mCellStiffness(0,0)=c*(1.0-v);

    if( paramList.isType<double>("Pressure Scaling") ){
      mPressureScaling = paramList.get<double>("Pressure Scaling");
    } else {
      mPressureScaling = k / (3.0*(1.0-2.0*v));
    }
}
/******************************************************************************/
template<>
::Plato::IsotropicLinearElasticMaterial<2>::
IsotropicLinearElasticMaterial(const Teuchos::ParameterList& paramList) :
LinearElasticMaterial<2>(paramList)
/******************************************************************************/
{
    mPoissonsRatio = paramList.get<double>("Poissons Ratio");
    mYoungsModulus = paramList.get<double>("Youngs Modulus");
    auto v = mPoissonsRatio;
    auto k = mYoungsModulus;
    auto c = k/((1.0+v)*(1.0-2.0*v));

    mCellStiffness(0,0)=c*(1.0-v); mCellStiffness(0,1)=c*v;
    mCellStiffness(1,0)=c*v;       mCellStiffness(1,1)=c*(1.0-v);
    mCellStiffness(2,2)=1.0/2.0*c*(1.0-2.0*v);

    if( paramList.isType<double>("Pressure Scaling") ){
      mPressureScaling = paramList.get<double>("Pressure Scaling");
    } else {
      mPressureScaling = k / (3.0*(1.0-2.0*v));
    }
}
/******************************************************************************/
template<>
::Plato::IsotropicLinearElasticMaterial<3>::
IsotropicLinearElasticMaterial(const Teuchos::ParameterList& paramList) :
LinearElasticMaterial<3>(paramList)
/******************************************************************************/
{
    mPoissonsRatio = paramList.get<double>("Poissons Ratio");
    mYoungsModulus = paramList.get<double>("Youngs Modulus");
    auto v = mPoissonsRatio;
    auto k = mYoungsModulus;
    auto c = k/((1.0+v)*(1.0-2.0*v));

    mCellStiffness(0,0)=c*(1.0-v); mCellStiffness(0,1)=c*v;       mCellStiffness(0,2)=c*v;
    mCellStiffness(1,0)=c*v;       mCellStiffness(1,1)=c*(1.0-v); mCellStiffness(1,2)=c*v;
    mCellStiffness(2,0)=c*v;       mCellStiffness(2,1)=c*v;       mCellStiffness(2,2)=c*(1.0-v);
    mCellStiffness(3,3)=1.0/2.0*c*(1.0-2.0*v);
    mCellStiffness(4,4)=1.0/2.0*c*(1.0-2.0*v);
    mCellStiffness(5,5)=1.0/2.0*c*(1.0-2.0*v);

    if( paramList.isType<double>("Pressure Scaling") ){
      mPressureScaling = paramList.get<double>("Pressure Scaling");
    } else {
      mPressureScaling = k / (3.0*(1.0-2.0*v));
    }

}

/******************************************************************************/
template<>
::Plato::IsotropicLinearElasticMaterial<1>::
IsotropicLinearElasticMaterial(const Plato::Scalar & aYoungsModulus, const Plato::Scalar & aPoissonsRatio) :
    LinearElasticMaterial<1>(),
    mPoissonsRatio(aPoissonsRatio),
    mYoungsModulus(aYoungsModulus)
/******************************************************************************/
{
    auto v = mPoissonsRatio;
    auto k = mYoungsModulus;
    auto c = k/((1.0+v)*(1.0-2.0*v));
    mCellStiffness(0,0)=c*(1.0-v);
}

/******************************************************************************/
template<>
::Plato::IsotropicLinearElasticMaterial<2>::
IsotropicLinearElasticMaterial(const Plato::Scalar & aYoungsModulus, const Plato::Scalar & aPoissonsRatio) :
    LinearElasticMaterial<2>(),
    mPoissonsRatio(aPoissonsRatio),
    mYoungsModulus(aYoungsModulus)
/******************************************************************************/
{
    auto v = mPoissonsRatio;
    auto k = mYoungsModulus;
    auto c = k/((1.0+v)*(1.0-2.0*v));
    mCellStiffness(0,0)=c*(1.0-v); mCellStiffness(0,1)=c*v;
    mCellStiffness(1,0)=c*v;       mCellStiffness(1,1)=c*(1.0-v);
    mCellStiffness(2,2)=1.0/2.0*c*(1.0-2.0*v);
}
/******************************************************************************/
template<>
::Plato::IsotropicLinearElasticMaterial<3>::
IsotropicLinearElasticMaterial(const Plato::Scalar & aYoungsModulus, const Plato::Scalar & aPoissonsRatio) :
    LinearElasticMaterial<3>(),
    mPoissonsRatio(aPoissonsRatio),
    mYoungsModulus(aYoungsModulus)
/******************************************************************************/
{
    auto v = mPoissonsRatio;
    auto k = mYoungsModulus;
    auto c = k/((1.0+v)*(1.0-2.0*v));
    mCellStiffness(0,0)=c*(1.0-v); mCellStiffness(0,1)=c*v;       mCellStiffness(0,2)=c*v;
    mCellStiffness(1,0)=c*v;       mCellStiffness(1,1)=c*(1.0-v); mCellStiffness(1,2)=c*v;
    mCellStiffness(2,0)=c*v;       mCellStiffness(2,1)=c*v;       mCellStiffness(2,2)=c*(1.0-v);
    mCellStiffness(3,3)=1.0/2.0*c*(1.0-2.0*v);
    mCellStiffness(4,4)=1.0/2.0*c*(1.0-2.0*v);
    mCellStiffness(5,5)=1.0/2.0*c*(1.0-2.0*v);
}

} // namespace Plato 
