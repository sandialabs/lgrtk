/*
 * HyperbolicTangentProjection.hpp
 *
 *  Created on: Apr 23, 2018
 */

#ifndef HYPERBOLICTANGENTPROJECTION_HPP_
#define HYPERBOLICTANGENTPROJECTION_HPP_

#include <cmath>

#include <Teuchos_ParameterList.hpp>

#include "plato/PlatoStaticsTypes.hpp"

namespace Plato
{

/****************************************************************************//**
 *
 * \brief Hyperbolic tangent projection class.
 *
 * Apply hyperbolic tangent projection, /f$H_{\eta}(\rho)/f$, where /f$\eta/f$
 * denotes a threshold level on the density /f$\rho/f$ and /f$\beta\geq{0}/f$
 * dictates the curvature of the regularization. The hyperbolic projection
 * approaches the Heaviside function as /f$\beta\rightarrow{0}/f$.
 *
 * The hyperbolic tangent projection is given by:
 *
 *    /f$H_{\eta}(\rho) = \frac{\tanh(\beta\eta) + \tanh(\beta(\rho - \eta))}
 *    {\tanh(\beta\eta) + \tanh(\beta(1 - \eta))}/f$
 *
********************************************************************************/
class HyperbolicTangentProjection
{
public:
    /****************************************************************************//**
     *
     * \brief Constructor
     *
     * Input Arguments:
     *
     * \param aBeta  dictates the curvature of the regularization
     * \param aEta   threshold on controls
     *
    *********************************************************************************/
    explicit HyperbolicTangentProjection(Plato::Scalar aBeta = 10, Plato::Scalar aEta = 0.5) :
        mEta(aEta),
        mBeta(aBeta),
        mDenominator(1),
        mBetaTimesEta(aBeta*aEta),
        mBetaTimesOneMinusEta(aBeta*(static_cast<Plato::Scalar>(1)-aEta))
    {
        mDenominator = std::tanh(mBetaTimesEta) + std::tanh(mBetaTimesOneMinusEta);
    }

    /****************************************************************************//**
     *
     * \brief Constructor
     *
     * Input Arguments:
     *
     * \param aPramList  Teuchos parameter list with parameters that control the
     *   behavior of the hyperbolic projection operator
     *
    *********************************************************************************/
    explicit HyperbolicTangentProjection(Teuchos::ParameterList & aPramList) :
        mEta(aPramList.get<Plato::Scalar>("Eta", 0.5)),
        mBeta(aPramList.get<Plato::Scalar>("Beta", 10)),
        mDenominator(1),
        mBetaTimesEta(mBeta*mEta),
        mBetaTimesOneMinusEta(mBeta*(static_cast<Plato::Scalar>(1)-mEta))
    {
        mDenominator = std::tanh(mBetaTimesEta) + std::tanh(mBetaTimesOneMinusEta);
    }

    /****************************************************************************//**
     *
     * \brief Destructor
     *
    *********************************************************************************/
    virtual ~HyperbolicTangentProjection()
    {
    }

    //! Returns application of the hyperbolic tangent projection to input scalar.
    template<typename ScalarType>
    KOKKOS_INLINE_FUNCTION ScalarType apply(const ScalarType & aInput) const
    {
        ScalarType tValue = mBeta * (aInput - mEta);
        return ((std::tanh(mBetaTimesEta) + std::tanh(tValue)) / mDenominator);
    }

    // ! Sets curvature parameter \beta.
    KOKKOS_INLINE_FUNCTION void setCurvatureParameterBeta(const Plato::Scalar & aInput)
    {
        mBeta = aInput;
        mBetaTimesEta = mBeta * mEta;
        mBetaTimesOneMinusEta = mBeta * (static_cast<Plato::Scalar>(1) - mEta);
        mDenominator = std::tanh(mBetaTimesEta) + std::tanh(mBetaTimesOneMinusEta);
    }

    // ! Sets treshold level parameter \eta.
    KOKKOS_INLINE_FUNCTION void setTresholdLevelParameterEta(const Plato::Scalar & aInput)
    {
        mEta = aInput;
        mBetaTimesEta = mBeta * mEta;
        mBetaTimesOneMinusEta = mBeta * (static_cast<Plato::Scalar>(1) - mEta);
        mDenominator = std::tanh(mBetaTimesEta) + std::tanh(mBetaTimesOneMinusEta);
    }

    // ! Sets curvature parameter \beta and treshold level parameter \eta.
    KOKKOS_INLINE_FUNCTION void setProjectionParameters(const Plato::Scalar & aBeta, const Plato::Scalar & aEta)
    {
        mEta = aEta;
        mBeta = aBeta;
        mBetaTimesEta = mBeta * mEta;
        mBetaTimesOneMinusEta = mBeta * (static_cast<Plato::Scalar>(1) - mEta);
        mDenominator = std::tanh(mBetaTimesEta) + std::tanh(mBetaTimesOneMinusEta);
    }

private:
    Plato::Scalar mEta;
    Plato::Scalar mBeta;
    Plato::Scalar mDenominator;
    Plato::Scalar mBetaTimesEta;
    Plato::Scalar mBetaTimesOneMinusEta;
};
// class HyperbolicTangentProjection

} // namespace Plato

#endif /* HYPERBOLICTANGENTPROJECTION_HPP_ */
