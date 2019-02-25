#ifndef LINEARELECTROELASTICMATERIAL_HPP
#define LINEARELECTROELASTICMATERIAL_HPP

#include <Omega_h_matrix.hpp>
#include <Teuchos_ParameterList.hpp>

#include "plato/PlatoStaticsTypes.hpp"

namespace Plato {

/******************************************************************************/
/*!
  \brief Base class for Linear Electroelastic material models
*/
  template<int SpatialDim>
  class LinearElectroelasticMaterial
/******************************************************************************/
{
  protected:
    static constexpr auto m_numVoigtTerms = (SpatialDim == 3) ? 6 : 
                                           ((SpatialDim == 2) ? 3 :
                                          (((SpatialDim == 1) ? 1 : 0)));
    static_assert(m_numVoigtTerms, "SpatialDim must be 1, 2, or 3.");

    Omega_h::Matrix<m_numVoigtTerms,m_numVoigtTerms> m_cellStiffness;
    Omega_h::Matrix<SpatialDim, m_numVoigtTerms> m_cellPiezoelectricCoupling;
    Omega_h::Matrix<SpatialDim, SpatialDim> m_cellPermittivity;

    Plato::Scalar m_alpha;
  
  public:
    LinearElectroelasticMaterial();
    decltype(m_cellStiffness)             getStiffnessMatrix()    const {return m_cellStiffness;}
    decltype(m_cellPiezoelectricCoupling) getPiezoMatrix()        const {return m_cellPiezoelectricCoupling;}
    decltype(m_cellPermittivity)          getPermittivityMatrix() const {return m_cellPermittivity;}
    decltype(m_alpha)                     getAlpha()              const {return m_alpha;}
};

/******************************************************************************/
template<int SpatialDim>
LinearElectroelasticMaterial<SpatialDim>::
LinearElectroelasticMaterial()
/******************************************************************************/
{
  for(int i=0; i<m_numVoigtTerms; i++)
    for(int j=0; j<m_numVoigtTerms; j++)
      m_cellStiffness(i,j) = 0.0;

  for(int i=0; i<SpatialDim; i++)
    for(int j=0; j<m_numVoigtTerms; j++)
      m_cellPiezoelectricCoupling(i,j) = 0.0;

  for(int i=0; i<SpatialDim; i++)
    for(int j=0; j<SpatialDim; j++)
      m_cellPermittivity(i,j) = 0.0;

  m_alpha = 1.0;
}

/******************************************************************************/
/*!
  \brief Derived class for isotropic linear thermoelastic material model
*/
  template<int SpatialDim>
  class IsotropicLinearElectroelasticMaterial : public LinearElectroelasticMaterial<SpatialDim>
/******************************************************************************/
{
  public:
    IsotropicLinearElectroelasticMaterial(const Teuchos::ParameterList& paramList);
    virtual ~IsotropicLinearElectroelasticMaterial(){}
};
// class IsotropicLinearElectroelasticMaterial

/******************************************************************************/
/*!
  \brief Factory for creating material models
*/
  template<int SpatialDim>
  class ElectroelasticModelFactory
/******************************************************************************/
{
  public:
    ElectroelasticModelFactory(const Teuchos::ParameterList& paramList) : m_paramList(paramList) {}
    Teuchos::RCP<Plato::LinearElectroelasticMaterial<SpatialDim>> create();
  private:
    const Teuchos::ParameterList& m_paramList;
};
/******************************************************************************/
template<int SpatialDim>
Teuchos::RCP<LinearElectroelasticMaterial<SpatialDim>>
ElectroelasticModelFactory<SpatialDim>::create()
/******************************************************************************/
{
  auto modelParamList = m_paramList.get<Teuchos::ParameterList>("Material Model");

  if( modelParamList.isSublist("Isotropic Linear Electroelastic") ){
    return Teuchos::rcp(new Plato::IsotropicLinearElectroelasticMaterial<SpatialDim>(modelParamList.sublist("Isotropic Linear Electroelastic")));
  }
  return Teuchos::RCP<Plato::LinearElectroelasticMaterial<SpatialDim>>(nullptr);
}

} // namespace Plato

#endif
