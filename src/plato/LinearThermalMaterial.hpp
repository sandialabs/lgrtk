#ifndef LINEARTHERMALMATERIAL_HPP
#define LINEARTHERMALMATERIAL_HPP

#include <Omega_h_matrix.hpp>
#include <Teuchos_ParameterList.hpp>

#include "plato/PlatoStaticsTypes.hpp"

namespace Plato
{

/******************************************************************************/
/*!
 \brief Base class for Linear Thermal material models
 */
template<int SpatialDim>
class LinearThermalMaterial
/******************************************************************************/
{
protected:
    Plato::Scalar mCellDensity;
    Plato::Scalar mCellSpecificHeat;
    Omega_h::Matrix<SpatialDim, SpatialDim> mCellConductivity;

public:
    LinearThermalMaterial();
    Omega_h::Matrix<SpatialDim, SpatialDim> getConductivityMatrix() const
    {
        return mCellConductivity;
    }
    Plato::Scalar getMassDensity() const
    {
        return mCellDensity;
    }
    Plato::Scalar getSpecificHeat() const
    {
        return mCellSpecificHeat;
    }
};

/******************************************************************************/
template<int SpatialDim>
LinearThermalMaterial<SpatialDim>::LinearThermalMaterial() :
        mCellDensity(0.0),
        mCellSpecificHeat(0.0)
/******************************************************************************/
{
    for(Plato::OrdinalType tIndexI = 0; tIndexI < SpatialDim; tIndexI++)
    {
        for(Plato::OrdinalType tIndexJ = 0; tIndexJ < SpatialDim; tIndexJ++)
        {
            mCellConductivity(tIndexI, tIndexJ) = 0.0;
        }
    }
}

/******************************************************************************/
/*!
 \brief Derived class for isotropic linear elastic material model
 */
template<int SpatialDim>
class IsotropicLinearThermalMaterial : public LinearThermalMaterial<SpatialDim>
/******************************************************************************/
{
    using LinearThermalMaterial<SpatialDim>::mCellConductivity;
    using LinearThermalMaterial<SpatialDim>::mCellDensity;
    using LinearThermalMaterial<SpatialDim>::mCellSpecificHeat;
public:
    IsotropicLinearThermalMaterial(const Teuchos::ParameterList& aParamList) :
            LinearThermalMaterial<SpatialDim>()
    {
        auto tConductivityCoef = aParamList.get < Plato::Scalar > ("Conductivity Coefficient");
        for(Plato::OrdinalType tIndex = 0; tIndex < SpatialDim; tIndex++)
        {
            mCellConductivity(tIndex, tIndex) = tConductivityCoef;
        }

        if(aParamList.isType < Plato::Scalar > ("Mass Density"))
        {
            mCellDensity = aParamList.get < Plato::Scalar > ("Mass Density");
        }
        else
        {
            mCellDensity = 1.0;
        }
        if(aParamList.isType < Plato::Scalar > ("Specific Heat"))
        {
            mCellSpecificHeat = aParamList.get < Plato::Scalar > ("Specific Heat");
        }
        else
        {
            mCellSpecificHeat = 1.0;
        }
    }
};

/******************************************************************************/
/*!
 \brief Factory for creating material models
 */
template<int SpatialDim>
class ThermalModelFactory
/******************************************************************************/
{
public:
    ThermalModelFactory(const Teuchos::ParameterList& aParamList) :
            mParamList(aParamList)
    {
    }
    Teuchos::RCP<LinearThermalMaterial<SpatialDim>> create();
private:
    const Teuchos::ParameterList& mParamList;
};
/******************************************************************************/
template<int SpatialDim>
Teuchos::RCP<LinearThermalMaterial<SpatialDim>> ThermalModelFactory<SpatialDim>::create()
/******************************************************************************/
{
    auto tModelParamList = mParamList.get < Teuchos::ParameterList > ("Material Model");

    if(tModelParamList.isSublist("Isotropic Linear Thermal"))
    {
        return Teuchos::rcp(new IsotropicLinearThermalMaterial<SpatialDim>(tModelParamList.sublist("Isotropic Linear Thermal")));
    }
    return Teuchos::RCP < LinearThermalMaterial < SpatialDim >> (nullptr);
}

}

#endif
