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

struct LevelSetInitialCondition
{
  LevelSetInitialCondition(Plato::Scalar radius, Plato::Scalar length) : Nlobes(5), Rmid(radius), Rdelta(0), xCenter(0.5*length), yCenter(0.5*length)  {}
  const int Nlobes;
  const Plato::Scalar Rmid;
  const Plato::Scalar Rdelta;
  const Plato::Scalar xCenter;
  const Plato::Scalar yCenter;

  Plato::Scalar
  operator()(Plato::Scalar x, Plato::Scalar y, Plato::Scalar z) const
  {
    const Plato::Scalar r = std::sqrt(
        (x - xCenter) * (x - xCenter) + (y - yCenter) * (y - yCenter));
    const Plato::Scalar theta = atan2(y - yCenter, x - xCenter);
    const Plato::Scalar rSurf = Rmid + Rdelta * sin(Nlobes * theta);
    return r - rSurf;
  }
};

/******************************************************************************//**
 * @brief Cylinder geometry model class
**********************************************************************************/
template<typename ScalarType = double>
class LevelSetCylinderInBox : public Plato::GeometryModel<ScalarType>
{
public:
    static constexpr int SpatialDim = 3;

    /******************************************************************************//**
     * @brief Default constructor
     **********************************************************************************/
    explicit LevelSetCylinderInBox(const ScalarType aRadius, const ScalarType aLength, MPI_Comm aComm = MPI_COMM_WORLD) :
            mComm(aComm),
            mRadius(aRadius),
            mLength(aLength)
    {
        this->build_mesh();
        mWriter = Omega_h::vtk::Writer("LevelSetCylinderInBox", &mMesh, SpatialDim);
        write_mesh(mWriter, mMesh, mHamiltonJacobiFields, mTime);
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
    ScalarType area()
    {
        const ScalarType tArea = level_set_area(mMesh, mHamiltonJacobiFields, mInterfaceWidth);
        return (tArea);
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
     * @brief update parameters that define a cylinder.
     * @param aParam parameters that define a cylinder
    **********************************************************************************/
    void update(const std::map<std::string, ScalarType>& aParam)
    {
        assert(aParam.find("Configuration") != aParam.end());
        Plato::Configuration::type_t tConfiguration =
                static_cast<Plato::Configuration::type_t>(aParam.find("Configuration")->second);

        switch (tConfiguration)
        {
            case Plato::Configuration::INITIAL:
            {
                this->updateInitialConfiguration(aParam);
                break;
            }
            case Plato::Configuration::DYNAMIC:
            {
                this->updateDynamicConfiguration(aParam);
                break;
            }
            default:
            {
                std::abort();
            }
        }
    }

    void compute_arrival_time_for_assumed_burn_rate()
    {
      LevelSetInitialCondition initialCondition(mRadius, mLength);
      initialize_level_set(mMesh, mHamiltonJacobiFields, initialCondition);

      const Plato::Scalar maxSpeed = initialize_constant_speed(mMesh, mHamiltonJacobiFields, mAssumedBurnRate);
      const Plato::Scalar dx = mesh_minimum_length_scale<SpatialDim>(mMesh);
      const Plato::Scalar dtau = 0.2*dx / maxSpeed; // Units of time
      mInterfaceWidth = 1.5*dx / maxSpeed; // Should have same units as level set

      compute_arrival_time(mMesh, mHamiltonJacobiFields, mInterfaceWidth, dtau);
    }

private:
    /******************************************************************************//**
     * @brief update initial configuration
     * @param aParam parameters that define a cylinder
     **********************************************************************************/
    void updateInitialConfiguration(const std::map<std::string, ScalarType>& aParam)
    {
        assert(aParam.find("Radius") != aParam.end());
        mRadius = aParam.find("Radius")->second;
        if(aParam.find("Length") != aParam.end())
        {
            mLength = aParam.find("Length")->second;
        }

        this->compute_arrival_time_for_assumed_burn_rate();
    }

    /******************************************************************************//**
     * @brief update dynamic configuration
     * @param aParam parameters that define a cylinder
     **********************************************************************************/
    void updateDynamicConfiguration(const std::map<std::string, ScalarType>& aParam)
    {
        assert(aParam.find("BurnRate") != aParam.end());
        const ScalarType tBurnRate = aParam.find("BurnRate")->second;

        assert(aParam.find("DeltaTime") != aParam.end());
        const ScalarType tDeltaTime = aParam.find("DeltaTime")->second;

        const Plato::Scalar deltaPseudoTime = tBurnRate/mAssumedBurnRate * tDeltaTime;
        offset_level_set(mMesh, mHamiltonJacobiFields, -deltaPseudoTime);

        mTime += tDeltaTime;
        ++mStep;

        const size_t printInterval = 1000; // How often do you want to output mesh?
        if(mStep % printInterval == 0)
        {
            write_mesh(mWriter, mMesh, mHamiltonJacobiFields, mTime);
        }
    }

    /******************************************************************************//**
     * @brief Build bounding box and fields on computational mesh
    **********************************************************************************/
    void build_mesh()
    {
        auto tLibOmegaH = std::make_shared < Omega_h::Library > (nullptr, nullptr, mComm);
        const Plato::Scalar tLengthX = mLength;
        const Plato::Scalar tLengthY = mLength;
        const Plato::Scalar tLengthZ = mLength;
        const size_t tNumCellX = mNumCellsPerSide;
        const size_t tNumCellY = mNumCellsPerSide;
        const size_t tNumCellZ = mNumCellsPerSide;
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

private:
    MPI_Comm mComm;
    ProblemFields<SpatialDim> mHamiltonJacobiFields;
    Omega_h::Mesh mMesh;
    Omega_h::vtk::Writer mWriter;
    size_t mNumCellsPerSide = 64;
    ScalarType mRadius;
    ScalarType mLength;
    ScalarType mInterfaceWidth;
    ScalarType mAssumedBurnRate = 1.0;
    ScalarType mTime = 0.0;
    size_t mStep = 0;
};
// class LevelSetCylinderInBox

} // namespace Plato
