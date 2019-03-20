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
 * @brief Cylinder geometry model class
**********************************************************************************/
class LevelSetOnExternalMesh : public Plato::GeometryModel
{
public:
    static constexpr int mSpatialDim = 3;

    /******************************************************************************//**
     * @brief Default constructor
     * @param [in] aInputFileName name of input file with initial geometry and field data
     * @param [in] aComm MPI communicator
     **********************************************************************************/
    explicit LevelSetOnExternalMesh(const std::string & aInputFileName, MPI_Comm aComm = MPI_COMM_WORLD) :
            mFilename(aInputFileName),
            mComm(aComm),
            mPropellantDensity(1744),
            mTimes()
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
    Plato::Scalar referencMassProductionRate() override
    {
        const Plato::Scalar tRefMassProdRate =
                mPropellantDensity * level_set_volume_rate_of_change(mMesh, mHamiltonJacobiFields, mInterfaceWidth);
        return tRefMassProdRate;
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
        this->updateLevelSetCylinder(aDeltaTime, aBurnRateMultiplier);
    }

    /******************************************************************************//**
     * @brief Update immersed geometry
     * @param [in] aParam optimization parameters
     **********************************************************************************/
    void updateGeometry(const Plato::ProblemParams & aParam) override
    {
    }

    /******************************************************************************//**
     * @brief Initialize level set cylinder
     * @param [in] aParam parameters associated with the geometry and fields
     **********************************************************************************/
    void initialize(const Plato::ProblemParams & aParam) override
    {
        mPropellantDensity = aParam.mPropellantDensity;
    }

    /******************************************************************************//**
     * @brief Output geometry and field data
     * @param [in] aOutput output flag (true = output, false = do not output)
    **********************************************************************************/
    void output(bool aOutput = false) override
    {
        if(aOutput == true)
        {
            this->outputLevelSet();
        }
    }

    /******************************************************************************//**
     * @brief Read level set based geometry from file
     **********************************************************************************/
    void initialize()
    {
        this->readMesh();
        this->cacheData();
        reinitialize_level_set(mMesh, mHamiltonJacobiFields, 0.0, mInterfaceWidth, mReIninitializationDeltaTime);
        mWriter = Omega_h::vtk::Writer("LevelSetOnExternalMesh", &mMesh, mSpatialDim);
        write_mesh(mWriter, mMesh, mHamiltonJacobiFields, mTime);
    }

private:
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
    void outputLevelSet()
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
    void updateLevelSetCylinder(const Plato::Scalar aDeltaTime, const Plato::Scalar aBurnRateMultiplier)
    {
        evolve_level_set(mMesh, mHamiltonJacobiFields, mInterfaceWidth, aBurnRateMultiplier*aDeltaTime);
        mTime += aDeltaTime;
        mTimes.push_back(mTime);

        reinitialize_level_set(mMesh, mHamiltonJacobiFields, mTime, mInterfaceWidth, mReIninitializationDeltaTime);

        ++mStep;
        this->cacheData();
    }

    /******************************************************************************//**
     * @brief Build bounding box and fields on computational mesh
     **********************************************************************************/
    void readMesh()
    {
    	auto tLibOmegaH = std::make_shared < Omega_h::Library > (nullptr, nullptr, mComm);
    	Omega_h::read_mesh_file(mFilename, tLibOmegaH->world());

        declare_fields(mMesh, mHamiltonJacobiFields);

        // TODO: Copy levelset and burn rate fields to the ones expected by H-J routines

        mDeltaX = mesh_minimum_length_scale<mSpatialDim>(mMesh);
        mInterfaceWidth = static_cast<Plato::Scalar>(1.5) * mDeltaX; // Should have same units as level set
        mReIninitializationDeltaTime = 0.2 * mDeltaX;
    }

private:
    std::string mFilename;
    MPI_Comm mComm;
    ProblemFields<mSpatialDim> mHamiltonJacobiFields;
    Omega_h::Mesh mMesh;
    Omega_h::vtk::Writer mWriter;
    Plato::Scalar mPropellantDensity = 0.0;
    Plato::Scalar mInterfaceWidth = 0.0;
    Plato::Scalar mReIninitializationDeltaTime = 0.0;
    Plato::Scalar mTime = 0.0;
    size_t mStep = 0;
    Plato::Scalar mDeltaX = 0.0;
    std::vector<Plato::Scalar> mTimes;
};
// class LevelSetOnExternalMesh

}// namespace Plato
