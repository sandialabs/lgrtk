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
    static constexpr auto m_numVoigtTerms = (SpatialDim == 3) ? 6 : 
                                           ((SpatialDim == 2) ? 3 :
                                          (((SpatialDim == 1) ? 1 : 0)));
    static_assert(m_numVoigtTerms, "SpatialDim must be 1, 2, or 3.");

    Omega_h::Matrix<m_numVoigtTerms,m_numVoigtTerms> m_cellStiffness;
  
  public:
    LinearElasticMaterial();
    Omega_h::Matrix<m_numVoigtTerms,m_numVoigtTerms> getStiffnessMatrix() const {return m_cellStiffness;}
};

/******************************************************************************/
template<int SpatialDim>
LinearElasticMaterial<SpatialDim>::
LinearElasticMaterial()
/******************************************************************************/
{
  for(int i=0; i<m_numVoigtTerms; i++)
    for(int j=0; j<m_numVoigtTerms; j++)
      m_cellStiffness(i,j) = 0.0;
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
    Plato::Scalar m_poissonsRatio;
    Plato::Scalar m_youngsModulus;
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
    ElasticModelFactory(const Teuchos::ParameterList& paramList) : m_paramList(paramList) {}
    Teuchos::RCP<Plato::LinearElasticMaterial<SpatialDim>> create();
  private:
    const Teuchos::ParameterList& m_paramList;
};
/******************************************************************************/
template<int SpatialDim>
Teuchos::RCP<LinearElasticMaterial<SpatialDim>>
ElasticModelFactory<SpatialDim>::create()
/******************************************************************************/
{
  auto modelParamList = m_paramList.get<Teuchos::ParameterList>("Material Model");

  if( modelParamList.isSublist("Isotropic Linear Elastic") ){
    return Teuchos::rcp(new Plato::IsotropicLinearElasticMaterial<SpatialDim>(modelParamList.sublist("Isotropic Linear Elastic")));
  }
  return Teuchos::RCP<Plato::LinearElasticMaterial<SpatialDim>>(nullptr);
}

} // namespace Plato

#endif
