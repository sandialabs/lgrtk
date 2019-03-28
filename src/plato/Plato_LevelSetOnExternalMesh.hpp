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
 * Plato_LevelSetOnExternalMesh.hpp
 *
 *  Created on: Aug 29, 2018
 */

#pragma once

#include "HamiltonJacobi.hpp"
#include "Omega_h_vector.hpp"
#include "Omega_h_build.hpp"
#include "Omega_h_class.hpp"
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

#include "plato/Plato_BuildMesh.hpp"
#include "plato/PlatoMathHelpers.hpp"
#include "plato/Plato_GeometryModel.hpp"

namespace Plato
{

/******************************************************************************//**
 * @brief Cylinder geometry model class
**********************************************************************************/
class LevelSetOnExternalMesh : public Plato::GeometryModel
{
public:
    static constexpr int mSpatialDim = 3;

    /******************************************************************************//**
     * @brief Default constructor
     * @param [in] aCoordsInputFile path to input file with coordinates information
     * @param [in] aConnInputFile path to input file with connectivity information
     * @param [in] aComm MPI communicator
     **********************************************************************************/
    explicit LevelSetOnExternalMesh(const std::string & aCoordsInputFile,
                                    const std::string & aConnInputFile,
                                    MPI_Comm aComm = MPI_COMM_WORLD) :
            mComm(aComm),
            mCoordsInputFile(aCoordsInputFile),
            mConnInputFile(aConnInputFile),
            mTimes(),
            mNodeFields(),
            mElementFields(),
            mStep(0),
            mNumNodes(0),
            mNumElems(0),
            mTime(0),
            mDeltaX(0),
            mInterfaceWidth(0),
            mPropellantDensity(1744),
            mSharpCornerAngle(Omega_h::PI/static_cast<Plato::Scalar>(4.0)),
            mReIninitializationDeltaTime(0),
            mOmegaLib(nullptr, nullptr, mComm),
            mMesh(&mOmegaLib)
    {
    }

    /******************************************************************************//**
     * @brief Destructor
     **********************************************************************************/
    virtual ~LevelSetOnExternalMesh()
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
    Plato::Scalar referenceMassProductionRate() override
    {
        const Plato::Scalar tRefMassProdRate =
                mPropellantDensity * level_set_volume_rate_of_change(mMesh, mHamiltonJacobiFields, mInterfaceWidth);
        return (tRefMassProdRate);
    }

    /******************************************************************************//**
     * @brief compute the gradient of a cylinder with respect to parameters that define geometry.
     * @param aOutput gradient with respect to the parameters that defined a geometry
     **********************************************************************************/
    void gradient(std::vector<Plato::Scalar>& aOutput) override { return; }

    /******************************************************************************//**
     * @brief Evolve geometry in time
     * @param [in] aDeltaTime time step
     * @param [in] aBurnRateMultiplier actual burn rate divided by the reference burn rate
     **********************************************************************************/
    void evolveGeometry(const Plato::Scalar aDeltaTime, const Plato::Scalar aBurnRateMultiplier) override
    {
        this->evolveImmersedGeometry(aDeltaTime, aBurnRateMultiplier);
    }

    /******************************************************************************//**
     * @brief Update initial immersed geometry
     * @param [in] aParam optimization parameters
     **********************************************************************************/
    void updateGeometry(const Plato::ProblemParams & aParam) override { return; }

    /******************************************************************************//**
     * @brief Set initial level set based geometry configuration
     * @param [in] aParam parameters associated with the geometry and fields
     **********************************************************************************/
    void initialize(const Plato::ProblemParams & aParam) override
    {
        mPropellantDensity = aParam.mPropellantDensity;
        mHamiltonJacobiFields.mNumTimeSteps = aParam.mNumTimeSteps;
        this->build();
        declare_fields(mMesh, mHamiltonJacobiFields);
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
     * @brief Read node-based level set field from text file
     * @param [in] aInputFile path to input file
    **********************************************************************************/
    void readNodalLevelSet(const std::string & aInputFile)
    {
        auto tNumNodes = mMesh.nverts();
        auto tReadNodeField = Plato::read_data(aInputFile.c_str(), tNumNodes);
        auto tNodeField = Plato::transform(tReadNodeField);
        auto tHostNodeField = Kokkos::create_mirror(mHamiltonJacobiFields.mLevelSet);
        for(Plato::OrdinalType tIndex = 0; tIndex < tNumNodes; tIndex++)
        {
            tHostNodeField(tIndex, mHamiltonJacobiFields.mCurrentState) = tNodeField[tIndex];
        }
        Kokkos::deep_copy(mHamiltonJacobiFields.mLevelSet, tHostNodeField);

        this->cacheData();
        reinitialize_level_set(mMesh, mHamiltonJacobiFields, 0.0, mInterfaceWidth, mReIninitializationDeltaTime);
    }

    /******************************************************************************//**
     * @brief Read element-based level set field from text file
     * @param [in] aInputFile path to input file
    **********************************************************************************/
    void readElementBurnRate(const std::string & aInputFile)
    {
        auto tNumElems = mMesh.nelems();
        auto tReadElemField = Plato::read_data(aInputFile.c_str(), tNumElems);
        auto tElemField = Plato::transform(tReadElemField);
        auto tHostElemField = Kokkos::create_mirror(mHamiltonJacobiFields.mElementSpeed);
        assert(tHostElemField.size() == tNumElems);
        for(Plato::OrdinalType tIndex = 0; tIndex < tNumElems; tIndex++)
        {
            tHostElemField(tIndex) = tElemField[tIndex];
        }
        Kokkos::deep_copy(mHamiltonJacobiFields.mElementSpeed, tHostElemField);
        mHamiltonJacobiFields.mUseElementSpeed = true;
    }

private:
    /******************************************************************************//**
     * @brief Build computational geometry (i.e. mesh) from text files
    **********************************************************************************/
    void build()
    {
        Plato::build_mesh_from_text_files<mSpatialDim>(mConnInputFile, mCoordsInputFile, mSharpCornerAngle, mMesh);
        mWriter = Omega_h::vtk::Writer("output", &mMesh, mSpatialDim);

        mDeltaX = mesh_minimum_length_scale<mSpatialDim>(mMesh);
        mInterfaceWidth = static_cast<Plato::Scalar>(1.5) * mDeltaX; // Should have same units as level set
        mReIninitializationDeltaTime = 0.2 * mDeltaX;
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
     * @brief Update immersed cylinder
     * @param [in] aParam optimization parameters
     **********************************************************************************/
    void evolveImmersedGeometry(const Plato::Scalar aDeltaTime, const Plato::Scalar aBurnRateMultiplier)
    {
        evolve_level_set(mMesh, mHamiltonJacobiFields, mInterfaceWidth, aBurnRateMultiplier*aDeltaTime);
        mTime += aDeltaTime;
        mTimes.push_back(mTime);

        reinitialize_level_set(mMesh, mHamiltonJacobiFields, mTime, mInterfaceWidth, mReIninitializationDeltaTime);

        ++mStep;
        this->cacheData();
    }

private:
    MPI_Comm mComm;

    std::string mConnInputFile;
    std::string mCoordsInputFile;
    std::vector<Plato::Scalar> mTimes;

    std::map<std::string, Omega_h::HostWrite<Plato::Scalar>> mNodeFields;
    std::map<std::string, Omega_h::HostWrite<Plato::Scalar>> mElementFields;

    size_t mStep;
    size_t mNumNodes;
    size_t mNumElems;

    Plato::Scalar mTime;
    Plato::Scalar mDeltaX;
    Plato::Scalar mInterfaceWidth;
    Plato::Scalar mPropellantDensity;
    Plato::Scalar mSharpCornerAngle;
    Plato::Scalar mReIninitializationDeltaTime;

    Omega_h::Library mOmegaLib;
    Omega_h::Mesh mMesh;
    Omega_h::vtk::Writer mWriter;

    ProblemFields<mSpatialDim> mHamiltonJacobiFields;
};
// class LevelSetOnExternalMesh

}// namespace Plato
