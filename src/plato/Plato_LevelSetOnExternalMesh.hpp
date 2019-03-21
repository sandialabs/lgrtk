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
            this->outputLevelSetField();
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

    /******************************************************************************//**
     * @brief Build bounding box and fields on computational mesh from input files
     **********************************************************************************/
    void readMesh()
    {

        declare_fields(mMesh, mHamiltonJacobiFields);

        // TODO: Copy levelset and burn rate fields to the ones expected by H-J routines

        mDeltaX = mesh_minimum_length_scale<mSpatialDim>(mMesh);
        mInterfaceWidth = static_cast<Plato::Scalar>(1.5) * mDeltaX; // Should have same units as level set
        mReIninitializationDeltaTime = 0.2 * mDeltaX;
    }

    Omega_h::HostWrite<Plato::Scalar> readCoordinates()
    {
        Omega_h::Vector<mSpatialDim> tVector;
        std::vector<Omega_h::Vector<mSpatialDim>> tCoordinates;

        Plato::Scalar tValue = 0;
        Plato::OrdinalType tCount = 0;
        Plato::OrdinalType tIndex = 0;
        std::ifstream tInputFile(mCoordsInputFile, std::ios_base::in);
        while(tInputFile >> tValue)
        {
            tVector[tIndex] = tValue;
            tIndex++;

            tCount++;
            if(tCount % mSpatialDim == 0)
            {
                tIndex = 0;
                tCoordinates.push_back(tVector);
            }
        }

        mNumNodes = tCount / mSpatialDim;
        tInputFile.close();

        Omega_h::HostWrite<Plato::Scalar> tHostCoords(tCount);
        for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < mNumNodes; ++tNodeIndex)
        {
            for(Plato::OrdinalType tDimIndex = 0; tDimIndex < mSpatialDim; ++tDimIndex)
            {
                tHostCoords[tNodeIndex * mSpatialDim + tDimIndex] =
                        tCoordinates[static_cast<std::size_t>(tNodeIndex)][tDimIndex];
            }
        }

        return (tHostCoords);
    }

    Omega_h::HostWrite<Omega_h::LO> readConnectivity()
    {
        const Plato::OrdinalType tNumNodesPerCell = mSpatialDim + 1;
        Omega_h::Vector<tNumNodesPerCell> tVector;
        std::vector<Omega_h::Vector<tNumNodesPerCell>> tCoonectivity;

        Plato::OrdinalType tValue = 0;
        Plato::OrdinalType tCount = 0;
        Plato::OrdinalType tIndex = 0;
        std::ifstream tInputFile(mConnInputFile, std::ios_base::in);
        while(tInputFile >> tValue)
        {
            tVector[tIndex] = tValue - 1;
            tIndex++;

            tCount++;
            if(tCount % tNumNodesPerCell == 0)
            {
                tIndex = 0;
                tCoonectivity.push_back(tVector);
            }
        }

        mNumElems = tCount / tNumNodesPerCell;
        tInputFile.close();

        Omega_h::HostWrite<Omega_h::LO> tHostConn(tCount);
        for(Plato::OrdinalType tElemIndex = 0; tElemIndex < mNumElems; ++tElemIndex)
        {
            for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < tNumNodesPerCell; ++tNodeIndex)
            {
                tHostConn[tElemIndex * tNumNodesPerCell + tNodeIndex] = tCoonectivity[tElemIndex][tNodeIndex];
            }
        }

        return (tHostConn);
    }

    void buildMesh()
    {
        Omega_h::HostWrite<Omega_h::LO> tHostConn = this->readConnectivity();
        Omega_h::HostWrite<Plato::Scalar> tHostCoords = this->readCoordinates();
        auto tConnMap = Omega_h::Read<Omega_h::LO>(tHostConn.write());
        Omega_h::build_from_elems_and_coords(&mMesh, OMEGA_H_SIMPLEX, mSpatialDim, tConnMap, tHostCoords.write());
    }

private:
    MPI_Comm mComm;

    std::string mConnInputFile;
    std::string mCoordsInputFile;
    std::vector<Plato::Scalar> mTimes;

    size_t mStep = 0;
    size_t mNumNodes = 0;
    size_t mNumElems = 0;

    Plato::Scalar mTime = 0.0;
    Plato::Scalar mDeltaX = 0.0;
    Plato::Scalar mInterfaceWidth = 0.0;
    Plato::Scalar mPropellantDensity = 0.0;
    Plato::Scalar mReIninitializationDeltaTime = 0.0;

    Omega_h::Mesh mMesh;
    Omega_h::vtk::Writer mWriter;

    ProblemFields<mSpatialDim> mHamiltonJacobiFields;
};
// class LevelSetOnExternalMesh

}// namespace Plato
