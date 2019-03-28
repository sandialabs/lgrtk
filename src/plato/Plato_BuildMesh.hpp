/*
 * Plato_BuildMesh.hpp
 *
 *  Created on: Mar 27, 2019
 */

#pragma once

#include <Omega_h_class.hpp>
#include <Omega_h_build.hpp>
#include <Omega_h_array.hpp>

#include <array>
#include <vector>
#include <string>
#include <fstream>

namespace Plato
{

/******************************************************************************//**
 * @brief Read coordinates from text file on disk
 * @param [in] aCoordsInputFile path to file
 * @return 1D array with coordinates
**********************************************************************************/
template<Plato::OrdinalType SpatialDim>
inline Omega_h::HostWrite<Plato::Scalar> read_coordinates(const std::string & aCoordsInputFile)
{
    Omega_h::Vector<SpatialDim> tVector;
    std::vector<Omega_h::Vector<SpatialDim>> tCoordinates;

    Plato::Scalar tValue = 0;
    Plato::OrdinalType tCount = 0;
    Plato::OrdinalType tIndex = 0;
    std::ifstream tInputFile(aCoordsInputFile, std::ios_base::in);
    while(tInputFile >> tValue)
    {
        tVector[tIndex] = tValue;
        tIndex++;

        tCount++;
        if(tCount % SpatialDim == 0)
        {
            tIndex = 0;
            tCoordinates.push_back(tVector);
        }
    }

    Plato::OrdinalType tNumNodes = tCount / SpatialDim;
    tInputFile.close();

    Omega_h::HostWrite<Plato::Scalar> tHostCoords(tCount);
    for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < tNumNodes; ++tNodeIndex)
    {
        for(Plato::OrdinalType tDimIndex = 0; tDimIndex < SpatialDim; ++tDimIndex)
        {
            tHostCoords[tNodeIndex * SpatialDim + tDimIndex] =
                    tCoordinates[static_cast<std::size_t>(tNodeIndex)][tDimIndex];
        }
    }

    return (tHostCoords);
}
// function read_coordinates

/******************************************************************************//**
 * @brief Read connectivity from text file on disk
 * @param [in] aConnInputFile path to file
 * @return 1D array with connectivity
**********************************************************************************/
template<Plato::OrdinalType SpatialDim>
inline Omega_h::HostWrite<Omega_h::LO> read_connectivity(const std::string & aConnInputFile)
{
    const Plato::OrdinalType tNumNodesPerCell = SpatialDim + 1;
    std::array<Plato::OrdinalType, tNumNodesPerCell> tArray;
    std::vector<std::array<Plato::OrdinalType, tNumNodesPerCell>> tCoonectivity;

    Plato::OrdinalType tValue = 0;
    Plato::OrdinalType tCount = 0;
    Plato::OrdinalType tIndex = 0;
    std::ifstream tInputFile(aConnInputFile, std::ios_base::in);
    while(tInputFile >> tValue)
    {
        tArray[tIndex] = tValue;
        tIndex++;

        tCount++;
        if(tCount % tNumNodesPerCell == 0)
        {
            tIndex = 0;
            tCoonectivity.push_back(tArray);
        }
    }

    auto tNumElems = tCount / tNumNodesPerCell;
    tInputFile.close();

    Omega_h::HostWrite<Omega_h::LO> tHostConn(tCount);
    for(Plato::OrdinalType tElemIndex = 0; tElemIndex < tNumElems; ++tElemIndex)
    {
        for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < tNumNodesPerCell; ++tNodeIndex)
        {
            tHostConn[tElemIndex * tNumNodesPerCell + tNodeIndex] = tCoonectivity[tElemIndex][tNodeIndex];
        }
    }

    return (tHostConn);
}
// function read_connectivity

/******************************************************************************//**
 * @brief Build Omega_h mesh from text file on disk
 * @param [in] aConnInputFile path to text file with connectivity
 * @param [in] aCoordsInputFile path to text file with coordinates
 * @param [in] aSharpCornerAngle angle within intersecting planes
 * @param [out] aMesh Omega_h mesh database
 * @return 1D array with connectivity
**********************************************************************************/
template<Plato::OrdinalType SpatialDim>
inline void build_mesh_from_text_files(const std::string & aConnInputFile,
                                       const std::string & aCoordsInputFile,
                                       const Plato::Scalar & aSharpCornerAngle,
                                       Omega_h::Mesh & aMesh)
{
    auto tHostConn = Plato::read_connectivity<SpatialDim>(aConnInputFile);
    auto tHostCoords = Plato::read_coordinates<SpatialDim>(aCoordsInputFile);
    auto tConnMap = Omega_h::Read<Omega_h::LO>(tHostConn.write());
    Omega_h::build_from_elems_and_coords(&aMesh, OMEGA_H_SIMPLEX, SpatialDim, tConnMap, tHostCoords.write());
    Omega_h::classify_by_angles(&aMesh, aSharpCornerAngle);
}

/******************************************************************************//**
 * @brief Read data from text files
 * @param [in] aConnInputFile path to file
 * @return 1D array with connectivity
**********************************************************************************/
std::vector<Plato::Scalar> read_data(const std::string & aInputFile, const Plato::OrdinalType & aLength)
{
    std::vector<Plato::Scalar> tOutput(aLength);
    std::ifstream tInputFile(aInputFile, std::ios_base::in);

    Plato::Scalar tValue = 0;
    Plato::OrdinalType tIndex = 0;
    while(tInputFile >> tValue)
    {
        tOutput[tIndex] = tValue;
        tIndex++;
    }
    return (tOutput);
}
// function read_data

/******************************************************************************//**
 * @brief Transform standard vector into an omega_h array
 * @param [in] aInput standard vector
 * @param [in] aName vector name
 * @return omega_h array
**********************************************************************************/
inline Omega_h::HostWrite<Omega_h::Real> transform(const std::vector<Plato::Scalar> & aInput,
                                                   std::string const& aName = "")
{
    const Plato::OrdinalType tLength = aInput.size();
    Omega_h::HostWrite<Omega_h::Real> tOutput(tLength, aName.c_str());
    for(Plato::OrdinalType tIndex = 0; tIndex < tLength; tIndex++)
    {
        tOutput[tIndex] = aInput[tIndex];
    }
    return (tOutput);
}
// function transform

} // namespace Plato
