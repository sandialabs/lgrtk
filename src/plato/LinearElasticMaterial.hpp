#ifndef LINEARELASTICMATERIAL_HPP
#define LINEARELASTICMATERIAL_HPP

#include <Omega_h_matrix.hpp>
#include <Teuchos_ParameterList.hpp>

#include "plato/PlatoStaticsTypes.hpp"

namespace Plato {

/******************************************************************************/
/*!
  \brief Base class for Linear Elastic material models
*/
  template<int SpatialDim>
  class LinearElasticMaterial
/******************************************************************************/
{
  protected:
    static constexpr auto mNumVoigtTerms = (SpatialDim == 3) ? 6 : 
                                           ((SpatialDim == 2) ? 3 :
                                          (((SpatialDim == 1) ? 1 : 0)));
    static_assert(mNumVoigtTerms, "SpatialDim must be 1, 2, or 3.");

    Omega_h::Matrix<mNumVoigtTerms,mNumVoigtTerms> mCellStiffness;
    Omega_h::Vector<mNumVoigtTerms> mReferenceStrain;
    Plato::Scalar mPressureScaling;
  
  public:
    LinearElasticMaterial();
    LinearElasticMaterial(const Teuchos::ParameterList& paramList);
    decltype(mCellStiffness)   getStiffnessMatrix() const {return mCellStiffness;}
    decltype(mPressureScaling) getPressureScaling() const {return mPressureScaling;}
    decltype(mReferenceStrain) getReferenceStrain() const {return mReferenceStrain;}

  private:
    void initialize ();
};

/******************************************************************************/
template<int SpatialDim>
void LinearElasticMaterial<SpatialDim>::
initialize()
/******************************************************************************/
{
  for(int i=0; i<mNumVoigtTerms; i++)
    for(int j=0; j<mNumVoigtTerms; j++)
      mCellStiffness(i,j) = 0.0;

  mPressureScaling = 1.0;

  for(int i=0; i<mNumVoigtTerms; i++)
    mReferenceStrain(i) = 0.0;
}


/******************************************************************************/
template<int SpatialDim>
LinearElasticMaterial<SpatialDim>::
LinearElasticMaterial()
/******************************************************************************/
{
  initialize();
}

/******************************************************************************/
template<int SpatialDim>
LinearElasticMaterial<SpatialDim>::
LinearElasticMaterial(const Teuchos::ParameterList& paramList)
/******************************************************************************/

{
  initialize();

  if( paramList.isType<double>("e11") )  mReferenceStrain(0) = paramList.get<double>("e11");
  if( paramList.isType<double>("e22") )  mReferenceStrain(1) = paramList.get<double>("e22");
  if( paramList.isType<double>("e33") )  mReferenceStrain(2) = paramList.get<double>("e33");
  if( paramList.isType<double>("e23") )  mReferenceStrain(3) = paramList.get<double>("e23");
  if( paramList.isType<double>("e13") )  mReferenceStrain(4) = paramList.get<double>("e13");
  if( paramList.isType<double>("e12") )  mReferenceStrain(5) = paramList.get<double>("e12");
}

/******************************************************************************/
/*!
  \brief Derived class for isotropic linear elastic material model
*/
  template<int SpatialDim>
  class IsotropicLinearElasticMaterial : public LinearElasticMaterial<SpatialDim>
/******************************************************************************/
{
  public:
    IsotropicLinearElasticMaterial(const Teuchos::ParameterList& paramList);
    IsotropicLinearElasticMaterial(const Plato::Scalar & aYoungsModulus, const Plato::Scalar & aPoissonsRatio);
    virtual ~IsotropicLinearElasticMaterial(){}

  private:
    Plato::Scalar mPoissonsRatio;
    Plato::Scalar mYoungsModulus;
};
// class IsotropicLinearElasticMaterial

/******************************************************************************/
/*!
  \brief Factory for creating material models
*/
  template<int SpatialDim>
  class ElasticModelFactory
/******************************************************************************/
{
  public:
    ElasticModelFactory(const Teuchos::ParameterList& paramList) : mParamList(paramList) {}
    Teuchos::RCP<Plato::LinearElasticMaterial<SpatialDim>> create();
  private:
    const Teuchos::ParameterList& mParamList;
};
/******************************************************************************/
template<int SpatialDim>
Teuchos::RCP<LinearElasticMaterial<SpatialDim>>
ElasticModelFactory<SpatialDim>::create()
/******************************************************************************/
{
  auto modelParamList = mParamList.get<Teuchos::ParameterList>("Material Model");

  if( modelParamList.isSublist("Isotropic Linear Elastic") ){
    return Teuchos::rcp(new Plato::IsotropicLinearElasticMaterial<SpatialDim>(modelParamList.sublist("Isotropic Linear Elastic")));
  }
  return Teuchos::RCP<Plato::LinearElasticMaterial<SpatialDim>>(nullptr);
}

} // namespace Plato

#endif
