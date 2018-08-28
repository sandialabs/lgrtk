/*
 * HeavisideProjection.hpp
 *
 *  Created on: Apr 23, 2018
 */

#ifndef HEAVISIDEPROJECTION_HPP_
#define HEAVISIDEPROJECTION_HPP_

#include <cmath>

#include "plato/PlatoStaticsTypes.hpp"

namespace Plato
{

/******************************************************************************//**
 *
 * \brief Heaviside projection class.
 *
 * Apply Heaviside function projection /f$1 - H(1-\rho)/f$, where /f$\rho/f$
 * denotes the material density. The Heaviside function is defined as:
 *
 *   /f$H(\alpha) = 1 - \exp(-\beta*\alpha) + \alpha\exp(-\beta)/f$,
 *
 * where /f$\alpha = 1 - \rho/f$ and /f$\beta\geq{0}/f$ dictates the curvature
 * of the regularization.
 *
**********************************************************************************/
class HeavisideProjection
{
public:
    /****************************************************************************//**
     *
     * \brief Constructor
     *
     * Input Arguments:
     *
     * \param aBeta  dictates the curvature of the regularization
     *
    *********************************************************************************/
    explicit HeavisideProjection(Plato::Scalar aBeta = 10) :
        mBeta(aBeta),
        mExpBeta(std::exp(-aBeta))
    {
    }

    /****************************************************************************//**
     *
     * \brief Destructor
     *
    *********************************************************************************/
    virtual ~HeavisideProjection()
    {
    }

    //! Returns application of the Heaviside function projection to input scalar.
    template<typename ScalarType>
    KOKKOS_INLINE_FUNCTION ScalarType apply(ScalarType aInput) const
    {
        ScalarType tOneMinusRho = static_cast<Plato::Scalar>(1) - aInput;
        ScalarType tBetaTimesOneMinusRho = mBeta * tOneMinusRho;
        const ScalarType tOutput =
            static_cast<Plato::Scalar>(1) - std::exp(-tBetaTimesOneMinusRho) + (tOneMinusRho * mExpBeta);
        return (tOutput);
    }

    // ! Sets curvature parameter \beta.
    KOKKOS_INLINE_FUNCTION void setCurvatureParameterBeta(const Plato::Scalar & aInput)
    {
        mBeta = aInput;
        mExpBeta = std::exp(-aInput);
    }

private:
    Plato::Scalar mBeta;
    Plato::Scalar mExpBeta;
};
// class HeavisideProjection

} // namespace Plato

#endif /* HEAVISIDEPROJECTION_HPP_ */
