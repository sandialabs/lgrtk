/*
 * Plato_RocketMocks.hpp
 *
 *  Created on: Feb 20, 2019
 */

#pragma once

namespace Plato
{

namespace RocketMocks
{

/******************************************************************************//**
 * @brief Setup example with cylindrical geometry and a constant burn rate
 * @param [in] aRadius cylinder's radius
 * @param [in] aLength cylinder's length
 * @param [in] aInitialRadius initial chamber radius
 * @param [in] aRefBurnRate reference burn rate
 **********************************************************************************/
Plato::ProblemParams set_constant_burn_rate_problem(Plato::Scalar aMaxRadius,
                                                    Plato::Scalar aLength,
                                                    Plato::Scalar aInitialRadius,
                                                    Plato::Scalar aRefBurnRate)
{
    Plato::ProblemParams tParams;
    tParams.mGeometry.push_back(aMaxRadius);
    tParams.mGeometry.push_back(aLength);
    tParams.mGeometry.push_back(aInitialRadius);
    tParams.mRefBurnRate.push_back(aRefBurnRate);
    tParams.mRefBurnRate.push_back(0.0); /* reference burn rate slope with radius */
    return tParams;
}
// function setupConstantBurnRateCylinder

/******************************************************************************//**
 * @brief Setup example with cylindrical geometry and a constant burn rate
 * @param [in] aMaxRadius maximum cylinder's radius
 * @param [in] aLength cylinder's length
 * @param [in] aInitialRadius initial chamber radius
 * @param [in] aRefBurnRate reference burn rate
 * @param [in] aRefBurnRateSlopeWithRadius reference burn rate dependence on radius
 **********************************************************************************/
Plato::ProblemParams set_linear_burn_rate_problem(Plato::Scalar aMaxRadius,
                                                  Plato::Scalar aLength,
                                                  Plato::Scalar aInitialRadius,
                                                  Plato::Scalar aRefBurnRate,
                                                  Plato::Scalar aRefBurnRateSlopeWithRadius)
{
    Plato::ProblemParams tParams;
    tParams.mGeometry.push_back(aMaxRadius);
    tParams.mGeometry.push_back(aLength);
    tParams.mGeometry.push_back(aInitialRadius);
    tParams.mRefBurnRate.push_back(aRefBurnRate);
    tParams.mRefBurnRate.push_back(aRefBurnRateSlopeWithRadius);
    return tParams;
}
// function setupLinearBurnRateCylinder

} // namespace RocketMocks

} // namespace Plato
