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
 * @param [in] aRefBurnRate reference burn rate
**********************************************************************************/
Plato::ProblemParams setupConstantBurnRateCylinder(Plato::Scalar aRadius, Plato::Scalar aLength, Plato::Scalar aRefBurnRate)
{
        Plato::ProblemParams tParams;
        tParams.mGeometry.push_back(aRadius);
        tParams.mGeometry.push_back(aLength);
        tParams.mRefBurnRate.push_back(aRefBurnRate);
        return tParams;
}
// function setupConstantBurnRateCylinder

} // namespace RocketMocks

} // namespace Plato
