/*
//@HEADER
// *************************************************************************
//   Plato Engine v.1.0: Copyright 2018, National Technology & Engineering
//                    Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Sandia Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact the Plato team (plato3D-help@sandia.gov)
//
// *************************************************************************
//@HEADER
*/

/*
 * Plato_Cylinder.hpp
 *
 *  Created on: Aug 29, 2018
 */

#pragma once

#define _USE_MATH_DEFINES

#include <map>
#include <string>
#include <math.h>
#include <vector>
#include <cassert>
#include <cstddef>
#include <cstdlib>

#include "Plato_GeometryModel.hpp"

namespace Plato
{

/******************************************************************************//**
 * @brief Cylinder geometry model class
**********************************************************************************/
template<typename ScalarType = double>
class Cylinder : public Plato::GeometryModel<ScalarType>
{
public:
    /******************************************************************************//**
     * @brief Default constructor
    **********************************************************************************/
    explicit Cylinder(const ScalarType aRadius, const ScalarType aLength) :
            mRadius(aRadius),
            mLength(aLength)
    {
    }

    /******************************************************************************//**
     * @brief Destructor
    **********************************************************************************/
    virtual ~Cylinder()
    {
    }

    /******************************************************************************//**
     * @brief Return cylinder's radius
     * @return radius
    **********************************************************************************/
    ScalarType radius() const
    {
        return (mRadius);
    }

    /******************************************************************************//**
     * @brief Return cylinder's length
     * @return length
    **********************************************************************************/
    ScalarType length() const
    {
        return (mLength);
    }

    /******************************************************************************//**
     * @brief compute the area of the side of a cylinder.
    **********************************************************************************/
    ScalarType area()
    {
        const ScalarType tArea = static_cast<ScalarType>(2) * M_PI * mRadius * mLength;
        return (tArea);
    }

    /******************************************************************************//**
     * @brief compute the gradient with respect to geometric parameters
     * @param aOutput gradient
    **********************************************************************************/
    void gradient(std::vector<ScalarType>& aOutput)
    {
        assert(aOutput.size() == static_cast<size_t>(2));
        aOutput[0] = static_cast<ScalarType>(2) * M_PI * mLength;
        aOutput[1] = static_cast<ScalarType>(2) * M_PI * mRadius;
    }

    /******************************************************************************//**
     * @brief Initialize geometry
     * @param [in] aParam parameters associated with the geometry and fields on the geometry
    **********************************************************************************/
    void initialize(const Plato::ProblemParams & aParam)
    {
        mRadius = aParam.mGeometry[0];
        if(aParam.mGeometry.size() > static_cast<size_t>(1))
        {
            mLength = aParam.mGeometry[1];
        }
    }

    /******************************************************************************//**
     * @brief Update geometry
     * @param [in] aParam optimization parameters
    **********************************************************************************/
    void updateGeometry(const Plato::ProblemParams & aParam)
    {
        const ScalarType tDeltaTime = aParam.mTimeStep;
        const ScalarType tBurnRate = aParam.mBurnRate[0];
        mRadius += tBurnRate * tDeltaTime;
    }

private:
    ScalarType mRadius; /*!< cylinder's radius */
    ScalarType mLength; /*!< cylinder's length */
};
// class Cylinder

} // namespace Plato
