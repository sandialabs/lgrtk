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
 * Plato_LevelSetCylinderInBox.hpp
 *
 *  Created on: Aug 29, 2018
 */

#pragma once

#include "HamiltonJacobi.hpp"
#include "Omega_h_build.hpp"
#include "Omega_h_file.hpp"

#define _USE_MATH_DEFINES

#include <map>
#include <string>
#include <math.h>
#include <vector>
#include <memory>
#include <cassert>
#include <cstddef>
#include <cstdlib>

#include "Plato_GeometryModel.hpp"

namespace Plato
{

/******************************************************************************//**
 * @brief Initial conditions for the algebraic function used to represent the level set field.
**********************************************************************************/
struct LevelSetInitialCondition
{
    /******************************************************************************//**
     * @brief Constructor
     * @param [in] aRadius cylinder's radius
     * @param [in] aLength cylinder's length
    **********************************************************************************/
    LevelSetInitialCondition(const Plato::Scalar & aRadius, const Plato::Scalar & aMaxRadius) :
            mNlobes(5),
            mRmid(aRadius),
            mRdelta(0),
            mXCenter(aMaxRadius),
            mYCenter(aMaxRadius)
    {
    }

    const Plato::OrdinalType mNlobes;
    const Plato::Scalar mRmid;
    const Plato::Scalar mRdelta;
    const Plato::Scalar mXCenter;
    const Plato::Scalar mYCenter;

    /******************************************************************************//**
     * @brief Compute level set value
     * @param [in] aX x-coordinate
     * @param [in] aY y-coordinate
     * @param [in] aZ z-coordinate
     * @return level set value
    **********************************************************************************/
    Plato::Scalar operator()(const Plato::Scalar & aX, const Plato::Scalar & aY, const Plato::Scalar & aZ) const
    {
        const Plato::Scalar tRadius = std::sqrt((aX - mXCenter) * (aX - mXCenter) + (aY - mYCenter) * (aY - mYCenter));
        const Plato::Scalar tTheta = atan2(aY - mYCenter, aX - mXCenter);
        const Plato::Scalar tRSurf = mRmid + mRdelta * sin(mNlobes * tTheta);
        return tRadius - tRSurf;
    }
};
// struct LevelSetInitialCondition

/******************************************************************************//**
 * @brief Initial conditions for the algebraic function used to represent the burn rate field
**********************************************************************************/
struct BurnRateInitialCondition
{
    /******************************************************************************//**
     * @brief Constructor
     * @param [in] aRefBurnRate constant burn rate
    **********************************************************************************/
    BurnRateInitialCondition(const Plato::Scalar & aRefBurnRate, const Plato::Scalar & aRefBurnRateSlopeWithRadius, const Plato::Scalar & aMaxRadius) :
            mRefBurnRate(aRefBurnRate),
            mRefBurnRateSlopeWithRadius(aRefBurnRateSlopeWithRadius),
            mXCenter(aMaxRadius),
            mYCenter(aMaxRadius)
    {
    }

    const Plato::Scalar mRefBurnRate;
    const Plato::Scalar mRefBurnRateSlopeWithRadius;
    const Plato::Scalar mXCenter;
    const Plato::Scalar mYCenter;

    /******************************************************************************//**
     * @brief Compute burn rate
     * @param [in] aX x-coordinate
     * @param [in] aY y-coordinate
     * @param [in] aZ z-coordinate
     * @return burn rate value
    **********************************************************************************/
    Plato::Scalar operator()(const Plato::Scalar & aX, const Plato::Scalar & aY, const Plato::Scalar & aZ) const
    {
    	const Plato::Scalar tRadius = std::sqrt((aX - mXCenter) * (aX - mXCenter) + (aY - mYCenter) * (aY - mYCenter));
        return mRefBurnRate + tRadius * mRefBurnRateSlopeWithRadius;
    }
};
// struct LevelSetInitialCondition

/******************************************************************************//**
 * @brief Cylinder geometry model class
**********************************************************************************/
template<typename ScalarType = double>
class LevelSetCylinderInBox : public Plato::GeometryModel<ScalarType>
{
public:
    static constexpr int mSpatialDim = 3;

    /******************************************************************************//**
     * @brief Default constructor
     **********************************************************************************/
    explicit LevelSetCylinderInBox(MPI_Comm aComm = MPI_COMM_WORLD) :
            mComm(aComm)
    {
    }

    /******************************************************************************//**
     * @brief Destructor
     **********************************************************************************/
    virtual ~LevelSetCylinderInBox()
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
        const ScalarType tArea = level_set_area(mMesh, mHamiltonJacobiFields, mInterfaceWidth);
        return (tArea);
    }

    /******************************************************************************//**
     * @brief Compute the reference rate that gas mass is begin produced
     * @return mass production rate
     **********************************************************************************/
    ScalarType referencMassProductionRate()
    {
        const ScalarType tRefMassProdRate =
                mPropellantDensity * mSpeedScale * level_set_volume_rate_of_change(mMesh, mHamiltonJacobiFields, mInterfaceWidth);
        return tRefMassProdRate;
    }

    /******************************************************************************//**
     * @brief compute the gradient of a cylinder with respect to parameters that define geometry.
     * @param aOutput gradient with respect to the parameters that defined a geometry
     **********************************************************************************/
    void gradient(std::vector<ScalarType>& aOutput)
    {
        return;
    }

    /******************************************************************************//**
     * @brief Evolve geometry in time
     * @param [in] aDeltaTime time step
     * @param [in] aBurnRateMultiplier actual burn rate divided by the reference burn rate
     **********************************************************************************/
    virtual void evolveGeometry(const ScalarType aDeltaTime, const ScalarType aBurnRateMultiplier)
    {
        this->updateLevelSetCylinder(aDeltaTime, aBurnRateMultiplier);
    }

    /******************************************************************************//**
     * @brief Update immersed geometry
     * @param [in] aParam optimization parameters
     **********************************************************************************/
    void updateGeometry(const Plato::ProblemParams & aParam)
    {
        assert(aParam.mGeometry.size() == static_cast<size_t>(3));
        mMaxRadius = aParam.mGeometry[0];
        mLength = aParam.mGeometry[1];
        mRadius = aParam.mGeometry[2];
        assert(aParam.mRefBurnRate.size() == static_cast<size_t>(2));
        mRefBurnRate = aParam.mRefBurnRate[0];
        mRefBurnRateSlopeWithRadius = aParam.mRefBurnRate[1];
        mPropellantDensity = aParam.mPropellantDensity;

        this->initializeLevelSetCylinder();
    }

    /******************************************************************************//**
     * @brief Initialize level set cylinder
     * @param [in] aParam parameters associated with the geometry and fields
     **********************************************************************************/
    void initialize(const Plato::ProblemParams & aParam)
    {
        this->updateGeometry(aParam);
    }

private:
    /******************************************************************************//**
     * @brief Update immersed cylinder
     * @param [in] aParam optimization parameters
     **********************************************************************************/
    void updateLevelSetCylinder(const ScalarType aDeltaTime, const ScalarType aBurnRateMultiplier)
    {
        const Plato::Scalar tDeltaPseudoTime = aBurnRateMultiplier * mSpeedScale * aDeltaTime;
        offset_level_set(mMesh, mHamiltonJacobiFields, -tDeltaPseudoTime);
        mTime += aDeltaTime;
        ++mStep;
    }

    /******************************************************************************//**
     * @brief Initialize immersed cylinder
     **********************************************************************************/
    void initializeLevelSetCylinder()
    {
        if(mLength != mMeshLength || mMaxRadius != mMeshMaxRadius)
        {
            build_mesh(mMaxRadius, mLength);
            mWriter = Omega_h::vtk::Writer("LevelSetCylinderInBox", &mMesh, mSpatialDim);
            write_mesh(mWriter, mMesh, mHamiltonJacobiFields, mTime);
        }

        LevelSetInitialCondition tInitialCondition(mRadius, mMaxRadius);
        initialize_level_set(mMesh, mHamiltonJacobiFields, tInitialCondition);

        BurnRateInitialCondition tSpeedInitialCondition(mRefBurnRate, mRefBurnRateSlopeWithRadius, mMaxRadius);
        initialize_interface_speed(mMesh, mHamiltonJacobiFields, tSpeedInitialCondition);
        mSpeedScale = normalize_interface_speed(mMesh, mHamiltonJacobiFields);

        const Plato::Scalar tRelativeSpeed = 1.0;
        const Plato::Scalar tDeltaX = mesh_minimum_length_scale<mSpatialDim>(mMesh);
        const Plato::Scalar tDeltaTau = static_cast<Plato::Scalar>(0.2) * tDeltaX / tRelativeSpeed; // Units of time
        mInterfaceWidth = static_cast<Plato::Scalar>(1.5) * tDeltaX / tRelativeSpeed; // Should have same units as level set

        compute_arrival_time(mMesh, mHamiltonJacobiFields, mInterfaceWidth, tDeltaTau);
    }

    /******************************************************************************//**
     * @brief Build bounding box and fields on computational mesh
     **********************************************************************************/
    void build_mesh(const ScalarType aMeshMaxRadius, const ScalarType aMeshLength)
    {
        mMeshMaxRadius = aMeshMaxRadius;
        mMeshLength = aMeshLength;
        auto tLibOmegaH = std::make_shared < Omega_h::Library > (nullptr, nullptr, mComm);
        const Plato::Scalar tLengthX = 2.*mMeshMaxRadius;
        const Plato::Scalar tLengthY = 2.*mMeshMaxRadius;
        const Plato::Scalar tLengthZ = mMeshLength;
        const size_t tNumCellsPerSide = 64;
        const size_t tNumCellX = tNumCellsPerSide;
        const size_t tNumCellY = tNumCellsPerSide;
        const size_t tNumCellZ = tNumCellsPerSide;
        mMesh = Omega_h::build_box(tLibOmegaH->world(),
                                   OMEGA_H_SIMPLEX,
                                   tLengthX,
                                   tLengthY,
                                   tLengthZ,
                                   tNumCellX,
                                   tNumCellY,
                                   tNumCellZ);
        declare_fields(mMesh, mHamiltonJacobiFields);
    }

    /******************************************************************************//**
     * @brief Write level set field on mesh every N number of time steps
     **********************************************************************************/
    void outputLevelSetField()
    {
        const size_t tPrintInterval = 1000; // How often do you want to output mesh?
        if(mStep % tPrintInterval == 0)
        {
            write_mesh(mWriter, mMesh, mHamiltonJacobiFields, mTime);
        }
    }

private:
    MPI_Comm mComm;
    ProblemFields<mSpatialDim> mHamiltonJacobiFields;
    Omega_h::Mesh mMesh;
    Omega_h::vtk::Writer mWriter;
    ScalarType mMaxRadius = 0.0;
    ScalarType mLength = 0.0;
    ScalarType mRadius = 0.0;
    ScalarType mMeshLength = 0.0;
    ScalarType mMeshMaxRadius = 0.0;
    ScalarType mPropellantDensity = 0.0;
    ScalarType mRefBurnRate = 0.0;
    ScalarType mRefBurnRateSlopeWithRadius = 0.0;
    ScalarType mSpeedScale = 0.0;
    ScalarType mInterfaceWidth = 0.0;
    ScalarType mTime = 0.0;
    size_t mStep = 0;
};
// class LevelSetCylinderInBox

}// namespace Plato
