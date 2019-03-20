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
 * Plato_AlgebraicRocketModel.hpp
 *
 *  Created on: Aug 29, 2018
 */

#pragma once

#define _USE_MATH_DEFINES

#include <map>
#include <cmath>
#include <math.h>
#include <cstdio>
#include <memory>
#include <cassert>
#include <cstddef>

#include "Plato_GeometryModel.hpp"

namespace Plato
{

/******************************************************************************//**
 * @brief Data structure for rocket problem input parameters.
**********************************************************************************/
struct AlgebraicRocketInputs
{
    Plato::Scalar mAlpha;
    Plato::Scalar mDeltaTime;                // seconds
    Plato::Scalar mRefPressure;              // Pascal
    Plato::Scalar mTotalBurnTime;            // seconds
    Plato::Scalar mThroatDiameter;           // meters
    Plato::Scalar mNewtonTolerance;          // Pascal
    Plato::Scalar mAmbientPressure;          // Pascal
    Plato::Scalar mCharacteristicVelocity;   // meters/seconds

    size_t mNumTimeSteps;                 /*!< number of simulation time steps */
    size_t mMaxNumNewtonItr;              /*!< number of Newton iterations */

    /******************************************************************************//**
     * @brief Default constructor
    **********************************************************************************/
    AlgebraicRocketInputs() :
            mAlpha(0.38),
            mDeltaTime(0.1),
            mRefPressure(3.5e6),
            mTotalBurnTime(10),
            mThroatDiameter(0.04),
            mNewtonTolerance(1.e-8),
            mAmbientPressure(101.325),
            mCharacteristicVelocity(1554.5),
            mNumTimeSteps(mTotalBurnTime/mDeltaTime),
            mMaxNumNewtonItr(1000)
    {
    }
};
// struct AlgebraicRocketInputs

/******************************************************************************//**
 * @brief Design the rocket chamber to achieve desired QoI profile.
 *
 * Nomenclature:
 * QoI - quantity of interest
 * m - meters
 * sec - seconds
 * Pa - Pascal
 * kg - kilograms
 **********************************************************************************/
class AlgebraicRocketModel
{
public:
    /******************************************************************************//**
     * @brief Constructor
     * @param aInputs input parameters for simulation
     * @param aGeomModel geometry model used for the rocket chamber
     **********************************************************************************/
    explicit AlgebraicRocketModel(const Plato::AlgebraicRocketInputs& aInputs,
                                  const std::shared_ptr<Plato::GeometryModel>& aChamberGeomModel) :
            mPrint(true),
            mNumNewtonItr(0),
            mMaxNumNewtonItr(aInputs.mMaxNumNewtonItr),
            mRefPressure(aInputs.mRefPressure), // Pa
            mAlpha(aInputs.mAlpha),
            mThroatDiameter(aInputs.mThroatDiameter), // m
            mCharacteristicVelocity(aInputs.mCharacteristicVelocity), // m/sec
            mAmbientPressure(aInputs.mAmbientPressure), // Pa
            mDeltaTime(aInputs.mDeltaTime), // sec
            mTotalBurnTime(aInputs.mTotalBurnTime), // sec
            mNewtonTolerance(aInputs.mNewtonTolerance), // Pa
            mInvPrefAlpha(),
            mTimes(),
            mThrustProfile(),
            mPressureProfile(),
            mImmersedGeomModel(aChamberGeomModel)
    {
    }

    /******************************************************************************//**
     * @brief Destructor
    **********************************************************************************/
    ~AlgebraicRocketModel()
    {
    }

    /******************************************************************************//**
     * @brief Disables output to console.
    **********************************************************************************/
    void disableOutput()
    {
        mPrint = false;
    }

    /******************************************************************************//**
     * @brief set maximum number of iterations for chamber's total pressure calculation.
     * @param aInput maximum number of iterations
     **********************************************************************************/
    void setMaxNumIterations(const size_t& aInput)
    {
        mMaxNumNewtonItr = aInput;
    }

    /******************************************************************************//**
     * @brief set propellant's reference pressure.
     * @param aInput propellant's reference pressure
     **********************************************************************************/
    void setRefPressure(const Plato::Scalar& aInput)
    {
        mRefPressure = aInput;
    }

    /******************************************************************************//**
     * @brief set exponent in burn rate equation.
     * @param aInput burn rate exponent
     **********************************************************************************/
    void setBurnRateExponent(const Plato::Scalar& aInput)
    {
        mAlpha = aInput;
    }

    /******************************************************************************//**
     * @brief set throat diameter.
     * @param aInput throat diameter
     **********************************************************************************/
    void setThroatDiameter(const Plato::Scalar& aInput)
    {
        mThroatDiameter = aInput;
    }

    /******************************************************************************//**
     * @brief set characteristic velocity.
     * @param aInput characteristic velocity
     **********************************************************************************/
    void setCharacteristicVelocity(const Plato::Scalar& aInput)
    {
        mCharacteristicVelocity = aInput;
    }

    /******************************************************************************//**
     * @brief set ambient pressure.
     * @param aInput ambient pressure
     **********************************************************************************/
    void setAmbientPressure(const Plato::Scalar& aInput)
    {
        mAmbientPressure = aInput;
    }

    /******************************************************************************//**
     * @brief set burn time step.
     * @param aInput time step
     **********************************************************************************/
    void setBurnTimeStep(const Plato::Scalar& aInput)
    {
        mDeltaTime = aInput;
    }

    /******************************************************************************//**
     * @brief set total burn time.
     * @param aInput total burn time
     **********************************************************************************/
    void setTotalBurnTime(const Plato::Scalar& aInput)
    {
        mTotalBurnTime = aInput;
    }

    /******************************************************************************//**
     * @brief returns time steps.
     **********************************************************************************/
    std::vector<Plato::Scalar> getTimeProfile() const
    {
        return (mTimes);
    }

    /******************************************************************************//**
     * @brief returns thrust values for each time snapshot.
     **********************************************************************************/
    std::vector<Plato::Scalar> getThrustProfile() const
    {
        return (mThrustProfile);
    }

    /******************************************************************************//**
     * @brief returns pressure values for each time snapshot.
     **********************************************************************************/
    std::vector<Plato::Scalar> getPressuresProfile() const
    {
        return (mPressureProfile);
    }

    /******************************************************************************//**
     * @brief Initialize geometry and material properties given a new parameter set
     *        from the optimizer.
     * @param aParam Problem database
    **********************************************************************************/
    void initialize(Plato::ProblemParams & aParams)
    {
        aParams.mNumTimeSteps = mTotalBurnTime / mDeltaTime;
        mImmersedGeomModel->initialize(aParams);
    }

    /******************************************************************************//**
     * @brief Output geometry and field data
     * @param [in] aOutput output flag (true = output, false = do not output)
    **********************************************************************************/
    void output(bool aOutput = false)
    {
        if(aOutput == true)
        {
            mImmersedGeomModel->output(aOutput);
        }
    }

    /******************************************************************************//**
     * @brief compute thrust and pressure profiles given a simple algebraic model for a rocket.
    **********************************************************************************/
    void solve()
    {
        mTimes.clear();
        assert(mTimes.empty() == true);
        mThrustProfile.clear();
        assert(mThrustProfile.empty() == true);
        mPressureProfile.clear();
        assert(mPressureProfile.empty() == true);

        // circular chamber parameterization
        mInvPrefAlpha = static_cast<Plato::Scalar>(1.0) / std::pow(mRefPressure, mAlpha);
        Plato::Scalar tThroatArea = M_PI * mThroatDiameter * mThroatDiameter / static_cast<Plato::Scalar>(4.0);

        // initialize variables
        Plato::Scalar tTime = 0.0;
        Plato::Scalar tThrust = 0.0;
        Plato::Scalar tTotalPressure = mRefPressure; // initial guess

        bool tBurning = true;
        while(tBurning == true)
        {
            if(mPrint == true)
            {
                printf("Time = %f,\t Thrust = %f,\t Pressure = %f\n", tTime, tThrust, tTotalPressure);
            }

            mTimes.push_back(tTime);
            mThrustProfile.push_back(tThrust);
            mPressureProfile.push_back(tTotalPressure);
            Plato::Scalar tMassProductionRate = mImmersedGeomModel->referencMassProductionRate();

            tTotalPressure = this->newton(tMassProductionRate, tTotalPressure, tThroatArea);

            tThrust = static_cast<Plato::Scalar>(269.0) * static_cast<Plato::Scalar>(9.8)
                    * tThroatArea * (tTotalPressure - mAmbientPressure)
                    / mCharacteristicVelocity;

            Plato::Scalar tBurnRateMultiplier = std::pow(tTotalPressure, mAlpha) * mInvPrefAlpha;
            mImmersedGeomModel->evolveGeometry(mDeltaTime, tBurnRateMultiplier);
            tTime += mDeltaTime;

            tBurning = tTime + mNewtonTolerance < mTotalBurnTime;
        }
    }

private:
    /******************************************************************************//**
     * @brief Newton solver.
     * @param aRefMassProductionRate current production rate of gas by mass
     * @param aTotalPressure total pressure at current time step
     * @param aThroatArea current throat area
     **********************************************************************************/
    Plato::Scalar newton(const Plato::Scalar& aRefMassProductionRate, const Plato::Scalar& aTotalPressure, const Plato::Scalar& aThroatArea)
    {
        bool tDone = false;
        Plato::Scalar tNewTotalPressure = aTotalPressure;

        mNumNewtonItr = 0;
        while(tDone == false)
        {
            Plato::Scalar tMyResidualEvaluation = this->residual(aRefMassProductionRate, tNewTotalPressure, aThroatArea);
            Plato::Scalar tMyJacobianEvaluation = this->jacobian(aRefMassProductionRate, tNewTotalPressure, aThroatArea);
            Plato::Scalar tDeltaPressure = static_cast<Plato::Scalar>(-1.0) * tMyResidualEvaluation / tMyJacobianEvaluation;
            tNewTotalPressure += tDeltaPressure;

            mNumNewtonItr += static_cast<size_t>(1);
            tDone = std::abs(tDeltaPressure) < mNewtonTolerance || mNumNewtonItr > mMaxNumNewtonItr;
        }

        return (tNewTotalPressure);
    }

    /******************************************************************************//**
     * @brief Jacobian evaluation.
     * @param aRefMassProductionRate current production rate of gas by mass
     * @param aTotalPressure total pressure at current time step
     * @param aThroatArea current throat area
     **********************************************************************************/
    Plato::Scalar jacobian(const Plato::Scalar& aRefMassProductionRate, const Plato::Scalar& aTotalPressure, const Plato::Scalar& aThroatArea)
    {
        Plato::Scalar tPower = mAlpha - static_cast<Plato::Scalar>(1);
        Plato::Scalar tValue = aRefMassProductionRate * mAlpha * mInvPrefAlpha
                            * std::pow(aTotalPressure, tPower)
                            - aThroatArea / mCharacteristicVelocity;
        return tValue;
    }

    /******************************************************************************//**
     * @brief Residual evaluation.
     * @param aRefMassProductionRate current production rate of gas by mass
     * @param aTotalPressure total pressure at current time step
     * @param aThroatArea current throat area
     **********************************************************************************/
    Plato::Scalar residual(const Plato::Scalar& aRefMassProductionRate, const Plato::Scalar& aTotalPressure, const Plato::Scalar& aThroatArea)
    {
        Plato::Scalar tValue = aRefMassProductionRate * mInvPrefAlpha * std::pow(aTotalPressure, mAlpha)
                - aThroatArea * aTotalPressure / mCharacteristicVelocity;
        return tValue;
    }

private:
    bool mPrint;
    size_t mNumNewtonItr;
    size_t mMaxNumNewtonItr;

    Plato::Scalar mRefPressure; // Pa
    Plato::Scalar mAlpha;
    Plato::Scalar mThroatDiameter; // m
    Plato::Scalar mCharacteristicVelocity; // m/sec
    Plato::Scalar mAmbientPressure; // Pa
    Plato::Scalar mDeltaTime; // sec
    Plato::Scalar mTotalBurnTime; // sec
    Plato::Scalar mNewtonTolerance; // Pa

    Plato::Scalar mInvPrefAlpha;

    std::vector<Plato::Scalar> mTimes;
    std::vector<Plato::Scalar> mThrustProfile;
    std::vector<Plato::Scalar> mPressureProfile;

    std::shared_ptr<Plato::GeometryModel> mImmersedGeomModel;
};
// class AlgebraicRocketModel

} //namespace Plato
