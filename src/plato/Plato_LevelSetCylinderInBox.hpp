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
#include "plato/PlatoMathHelpers.hpp"

namespace Plato
{

/******************************************************************************//**
 * @brief Initial conditions for the algebraic function used to represent the level set field.
**********************************************************************************/
struct LevelSetInitialCondition
{
    /******************************************************************************//**
     * @brief Constructor
     * @param [in] aParams parameters associated with the geometry and fields
    **********************************************************************************/
    explicit LevelSetInitialCondition(const Plato::ProblemParams & aParams) :
            mNlobes(5),
            mRmid(aParams.mGeometry[2]),
            mRdelta(0),
            mXCenter(aParams.mGeometry[0]),
            mYCenter(aParams.mGeometry[0])
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
        const Plato::Scalar tOutput = tRadius - tRSurf;
        return (tOutput);
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
     * @param [in] aParams parameters associated with the geometry and fields
    **********************************************************************************/
    explicit BurnRateInitialCondition(const Plato::ProblemParams & aParams) :
            mRefBurnRate(aParams.mRefBurnRate[0]),
            mRefBurnRateSlopeWithRadius(aParams.mRefBurnRate[1]),
            mXCenter(aParams.mGeometry[0]),
            mYCenter(aParams.mGeometry[0])
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
class LevelSetCylinderInBox : public Plato::GeometryModel
{
public:
    static constexpr Plato::OrdinalType mSpatialDim = 3;

    /******************************************************************************//**
     * @brief Default constructor
     **********************************************************************************/
    explicit LevelSetCylinderInBox(MPI_Comm aComm = MPI_COMM_WORLD) :
            mComm(aComm),
            mTimes()
    {
    }

    /******************************************************************************//**
     * @brief Destructor
     **********************************************************************************/
    virtual ~LevelSetCylinderInBox()
    {
    }

    /******************************************************************************//**
     * @brief compute the area of the side of a cylinder.
     **********************************************************************************/
    Plato::Scalar area() override
    {
        const Plato::Scalar tArea = level_set_area(mMesh, mHamiltonJacobiFields, mInterfaceWidth);
        return (tArea);
    }

    /******************************************************************************//**
     * @brief Compute the reference rate that gas mass is begin produced
     * @return mass production rate
     **********************************************************************************/
    Plato::Scalar referencMassProductionRate() override
    {
        const Plato::Scalar tRefMassProdRate =
                mPropellantDensity * level_set_volume_rate_of_change(mMesh, mHamiltonJacobiFields, mInterfaceWidth);
        return tRefMassProdRate;
    }

    /******************************************************************************//**
     * @brief Output geometry and field data
     * @param [in] aOutput output flag (true = output, false = do not output)
     **********************************************************************************/
    void output(bool aOutput = false) override
    {
        if(aOutput == true)
        {
            this->outputLevelSetField();
        }
    }

    /******************************************************************************//**
     * @brief compute the gradient of a cylinder with respect to parameters that define geometry.
     * @param aOutput gradient with respect to the parameters that defined a geometry
     **********************************************************************************/
    void gradient(std::vector<Plato::Scalar>& aOutput) override
    {
        return;
    }

    /******************************************************************************//**
     * @brief Evolve geometry in time
     * @param [in] aDeltaTime time step
     * @param [in] aBurnRateMultiplier actual burn rate divided by the reference burn rate
     **********************************************************************************/
    void evolveGeometry(const Plato::Scalar aDeltaTime, const Plato::Scalar aBurnRateMultiplier) override
    {
        this->updateImmersedGeometry(aDeltaTime, aBurnRateMultiplier);
    }

    /******************************************************************************//**
     * @brief Update immersed geometry
     * @param [in] aParams optimization parameters
     **********************************************************************************/
    void updateGeometry(const Plato::ProblemParams & aParams) override
    {
        mPropellantDensity = aParams.mPropellantDensity;
        mHamiltonJacobiFields.mNumTimeSteps = aParams.mNumTimeSteps;
        this->initializeImmersedGeometry(aParams);
    }

    /******************************************************************************//**
     * @brief Initialize level set cylinder
     * @param [in] aParams parameters associated with the geometry and fields
     **********************************************************************************/
    void initialize(const Plato::ProblemParams & aParams) override
    {
        mTime = 0.;
        mTimes.clear();
        this->updateGeometry(aParams);
    }

private:
    /******************************************************************************//**
     * @brief Output level set time history to visualization file
    **********************************************************************************/
    void outputLevelSetField()
    {
        auto tNodeCount = mMesh.nverts();
        Kokkos::View<Omega_h::Real*> tOutput("into", tNodeCount);
        const Plato::OrdinalType tNumTimeSteps = mTimes.size();
        for(Plato::OrdinalType tIndex = 0; tIndex < tNumTimeSteps; tIndex++)
        {
            auto tSubView = Kokkos::subview(mHamiltonJacobiFields.mLevelSetHistory, Kokkos::ALL(), tIndex);
            Kokkos::deep_copy(tOutput, tSubView);
            mMesh.add_tag(Omega_h::VERT, "LevelSet", 1, Omega_h::Reals(Omega_h::Write<Omega_h::Real>(tOutput)));
            auto tTags = Omega_h::vtk::get_all_vtk_tags(&mMesh, mSpatialDim);
            mWriter.write(static_cast<Omega_h::Real>(mTimes[tIndex]), tTags);
        }
    }

    /******************************************************************************//**
     * @brief Cache level set field at this time snapshot
    **********************************************************************************/
    void cacheData()
    {
        auto tMyLevelSet = Kokkos::subview(mHamiltonJacobiFields.mLevelSet, Kokkos::ALL(), mHamiltonJacobiFields.mCurrentState);
        auto tMyOutput = Kokkos::subview(mHamiltonJacobiFields.mLevelSetHistory, Kokkos::ALL(), mStep);
        Plato::copy(tMyLevelSet, tMyOutput);
    }

    /******************************************************************************//**
     * @brief Update immersed cylinder
     * @param [in] aParams optimization parameters
     **********************************************************************************/
    void updateImmersedGeometry(const Plato::Scalar aDeltaTime, const Plato::Scalar aBurnRateMultiplier)
    {
        evolve_level_set(mMesh, mHamiltonJacobiFields, mInterfaceWidth, aBurnRateMultiplier*aDeltaTime);
        mTime += aDeltaTime;
        mTimes.push_back(mTime);

        reinitialize_level_set(mMesh, mHamiltonJacobiFields, mTime, mInterfaceWidth, mReIninitializationDeltaTime);

        ++mStep;
        this->cacheData();
    }

    /******************************************************************************//**
     * @brief Initialize immersed geometry
     * @param [in] aParams parameters associated with the geometry and fields
     **********************************************************************************/
    void initializeImmersedGeometry(const Plato::ProblemParams & aParams)
    {
        this->buildBoundingBox(aParams);

        LevelSetInitialCondition tInitialCondition(aParams);
        initialize_level_set(mMesh, mHamiltonJacobiFields, tInitialCondition);

        BurnRateInitialCondition tSpeedInitialCondition(aParams);
        initialize_interface_speed(mMesh, mHamiltonJacobiFields, tSpeedInitialCondition);

        reinitialize_level_set(mMesh, mHamiltonJacobiFields, 0.0, mInterfaceWidth, mReIninitializationDeltaTime);
    }

    /******************************************************************************//**
     * @brief Build bounding box and fields on computational mesh
     * @param [in] aMeshMaxRadius maximum radius for bounding box
     * @param [in] aMeshLength maximum bounding box length
     **********************************************************************************/
    void buildMesh(const Plato::Scalar & aMeshMaxRadius, const Plato::Scalar & aMeshLength)
    {
        mMeshMaxRadius = aMeshMaxRadius;
        mMeshLength = aMeshLength;
        const Plato::Scalar tLengthZ = mMeshLength;
        const Plato::Scalar tLengthX = static_cast<Plato::Scalar>(2) * mMeshMaxRadius;
        const Plato::Scalar tLengthY = static_cast<Plato::Scalar>(2) * mMeshMaxRadius;
        const size_t tNumCellsPerSide = 64;
        const size_t tNumCellX = tNumCellsPerSide;
        const size_t tNumCellY = tNumCellsPerSide;
        const size_t tNumCellZ = tNumCellsPerSide;
        auto tLibOmegaH = std::make_shared < Omega_h::Library > (nullptr, nullptr, mComm);
        mMesh = Omega_h::build_box(tLibOmegaH->world(),
                                   OMEGA_H_SIMPLEX,
                                   tLengthX,
                                   tLengthY,
                                   tLengthZ,
                                   tNumCellX,
                                   tNumCellY,
                                   tNumCellZ);
        declare_fields(mMesh, mHamiltonJacobiFields);

        mDeltaX = mesh_minimum_length_scale<mSpatialDim>(mMesh);
        mInterfaceWidth = static_cast<Plato::Scalar>(1.5) * mDeltaX; // Should have same units as level set
        mReIninitializationDeltaTime = static_cast<Plato::Scalar>(0.2) * mDeltaX;
    }

    /******************************************************************************//**
     * @brief Build bounding box
     * @param [in] aParams parameters associated with the geometry and fields
    **********************************************************************************/
    void buildBoundingBox(const Plato::ProblemParams & aParams)
    {
        if(mBuiltBoundingBox == false)
        {
            const Plato::Scalar tLength = aParams.mGeometry[1];
            const Plato::Scalar tMaxRadius = aParams.mGeometry[0];
            this->buildMesh(tMaxRadius, tLength);
            mBuiltBoundingBox = true;
            mWriter = Omega_h::vtk::Writer("LevelSetCylinderInBox", &mMesh, mSpatialDim);
            this->cacheData();
        }
    }

private:
    MPI_Comm mComm;
    bool mBuiltBoundingBox = false;
    ProblemFields<mSpatialDim> mHamiltonJacobiFields;
    Omega_h::Mesh mMesh;
    Omega_h::vtk::Writer mWriter;
    Plato::Scalar mMeshLength = 0.0;
    Plato::Scalar mMeshMaxRadius = 0.0;
    Plato::Scalar mPropellantDensity = 0.0;
    Plato::Scalar mRefBurnRate = 0.0;
    Plato::Scalar mRefBurnRateSlopeWithRadius = 0.0;
    Plato::Scalar mInterfaceWidth = 0.0;
    Plato::Scalar mReIninitializationDeltaTime = 0.0;
    Plato::Scalar mTime = 0.0;
    Plato::Scalar mDeltaX = 0.0;
    Plato::OrdinalType mStep = 0;
    std::vector<Plato::Scalar> mTimes;
};
// class LevelSetCylinderInBox

}// namespace Plato
