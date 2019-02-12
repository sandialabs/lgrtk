/*
 * Plato_VonMisesYield.hpp
 *
 *  Created on: Feb 10, 2019
 */

#pragma once

#include "plato/PlatoStaticsTypes.hpp"

namespace Plato
{

/******************************************************************************//**
 * @brief Compute Von Mises yield criterion
**********************************************************************************/
template<Plato::OrdinalType SpatialDim>
class VonMisesYield
{
public:
    /******************************************************************************//**
     * @brief Constructor
    **********************************************************************************/
    VonMisesYield(){}

    /******************************************************************************//**
     * @brief Destructor
    **********************************************************************************/
    ~VonMisesYield(){}

    /******************************************************************************//**
     * @brief Compute Von Mises yield criterion
     * @param [in] aCellOrdinal cell/element index
     * @param [in] aCauchyStress 2D container of cell Cauchy stresses
     * @param [out] aVonMisesStress 1D container of cell Von Mises yield stresses
    **********************************************************************************/
    template<typename Inputype, typename ResultType>
    DEVICE_TYPE inline void
    operator()(const Plato::OrdinalType & aCellOrdinal,
               const Plato::ScalarMultiVectorT<Inputype> & aCauchyStress,
               const Plato::ScalarVectorT<ResultType> & aVonMisesStress) const;
};
// class VonMisesYield

/******************************************************************************//**
 * @brief Von Mises yield criterion for 3D problems
**********************************************************************************/
template<>
template<typename Inputype, typename ResultType>
DEVICE_TYPE inline void
VonMisesYield<3>::operator()(const Plato::OrdinalType & aCellOrdinal,
        const Plato::ScalarMultiVectorT<Inputype> & aCauchyStress,
        const Plato::ScalarVectorT<ResultType> & aVonMisesStress) const
{
    Inputype tSigma11MinusSigma22 = aCauchyStress(aCellOrdinal, 0) - aCauchyStress(aCellOrdinal, 1);
    Inputype tSigma22MinusSigma33 = aCauchyStress(aCellOrdinal, 1) - aCauchyStress(aCellOrdinal, 2);
    Inputype tSigma33MinusSigma11 = aCauchyStress(aCellOrdinal, 2) - aCauchyStress(aCellOrdinal, 0);
    Inputype tPrincipalStressContribution = tSigma11MinusSigma22 * tSigma11MinusSigma22 +
            tSigma22MinusSigma33 * tSigma22MinusSigma33 + tSigma33MinusSigma11 * tSigma33MinusSigma11;

    Inputype tShearStressContribution = static_cast<Plato::Scalar>(3) *
            ( aCauchyStress(aCellOrdinal, 3) * aCauchyStress(aCellOrdinal, 3)
            + aCauchyStress(aCellOrdinal, 4) * aCauchyStress(aCellOrdinal, 4)
            + aCauchyStress(aCellOrdinal, 5) * aCauchyStress(aCellOrdinal, 5) );

    ResultType tVonMises = static_cast<Plato::Scalar>(0.5) * ( tPrincipalStressContribution + tShearStressContribution);
    aVonMisesStress(aCellOrdinal) = pow(tVonMises, static_cast<Plato::Scalar>(0.5));
}

/******************************************************************************//**
 * @brief Von Mises yield criterion for 2D problems (i.e. general plane stress)
**********************************************************************************/
template<>
template<typename Inputype, typename ResultType>
DEVICE_TYPE inline void
VonMisesYield<2>::operator()(const Plato::OrdinalType & aCellOrdinal,
        const Plato::ScalarMultiVectorT<Inputype> & aCauchyStress,
        const Plato::ScalarVectorT<ResultType> & aVonMisesStress) const
{
    Inputype tSigma11TimesSigma11 = aCauchyStress(aCellOrdinal, 0) * aCauchyStress(aCellOrdinal, 0);
    Inputype tSigma11TimesSigma22 = aCauchyStress(aCellOrdinal, 0) * aCauchyStress(aCellOrdinal, 1);
    Inputype tSigma22TimesSigma22 = aCauchyStress(aCellOrdinal, 1) * aCauchyStress(aCellOrdinal, 1);
    Inputype tSigma12TimesSigma12 = aCauchyStress(aCellOrdinal, 2) * aCauchyStress(aCellOrdinal, 2);

    ResultType tVonMises = tSigma11TimesSigma11 - tSigma11TimesSigma22 + tSigma22TimesSigma22 +
            static_cast<Plato::Scalar>(3) * tSigma12TimesSigma12;
    aVonMisesStress(aCellOrdinal) = pow(tVonMises, static_cast<Plato::Scalar>(0.5));
}

/******************************************************************************//**
 * @brief Von Mises yield criterion for 1D problems (i.e. uniaxial stress)
**********************************************************************************/
template<>
template<typename Inputype, typename ResultType>
DEVICE_TYPE inline void
VonMisesYield<1>::operator()(const Plato::OrdinalType & aCellOrdinal,
        const Plato::ScalarMultiVectorT<Inputype> & aCauchyStress,
        const Plato::ScalarVectorT<ResultType> & aVonMisesStress) const
{
    aVonMisesStress(aCellOrdinal) = aCauchyStress(aCellOrdinal, 0);
}

} // namespace Plato
