/*
//@HEADER
// ************************************************************************
//
//                        lgr v. 1.0
//              Copyright (2014) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
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
// 3. Neither the name of the Corporation nor the names of the
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
// Questions? Contact  Glen A. Hansen (gahanse@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#ifndef LGR_FIELD_DB_HPP
#define LGR_FIELD_DB_HPP

#include <map>
#include <sstream>
#include <string>

#include <Teuchos_TestForException.hpp>
#include <Kokkos_Macros.hpp>

namespace lgr {

template <typename FieldType>
class FieldDB : public std::map<std::string, FieldType> {
  FieldDB(){};
  ~FieldDB(){};
  FieldDB(const FieldDB<FieldType>&) = delete;
  FieldDB<FieldType>&       operator=(const FieldDB<FieldType>&) = delete;
  static FieldDB<FieldType> S;

 public:
  static FieldDB<FieldType>& Self() { return S; }

  FieldType& operator[](const std::string& keyName) {
    if (this->std::map<std::string, FieldType>::find(keyName) ==
        this->std::map<std::string, FieldType>::end()) {
      // create a new FieldType object (a Kokkos View, in fact), with label involving keyName
      std::ostringstream label;
      label << "FieldDB: " << keyName;
      using LayoutType = typename FieldType::array_layout;
      FieldType emptyContainer(label.str(), LayoutType());
      this->std::map<std::string, FieldType>::insert(
          std::make_pair(keyName, emptyContainer));
    }
    return this->std::map<std::string, FieldType>::operator[](keyName);
  }
};

template <typename T>
FieldDB<T> FieldDB<T>::S;

template <int SpatialDim>
void FieldDB_Finalize();

template <class ArrayType>
ArrayType safeFieldLookup(const std::string &key) {
  auto foundEntry = FieldDB<ArrayType>::Self().find(key);
  TEUCHOS_TEST_FOR_EXCEPTION(foundEntry == FieldDB<ArrayType>::Self().end(),
      std::invalid_argument, "Error: Field \"" << key << "\" not found in Fields DB.\n");
  return foundEntry->second;
}

template <class Fields>
typename Fields::geom_state_array_type Coordinates() {
  return safeFieldLookup<typename Fields::geom_state_array_type>(
      "spatial coordinates");
}

template <class Fields>
typename Fields::geom_state_array_type Velocity() {
  return safeFieldLookup<typename Fields::geom_state_array_type>("velocity");
}

template <class Fields>
typename Fields::geom_array_type Displacement() {
  return safeFieldLookup<typename Fields::geom_array_type>("displacement");
}

template <class Fields>
typename Fields::geom_array_type Acceleration() {
  return safeFieldLookup<typename Fields::geom_array_type>("acceleration");
}

template <class Fields>
typename Fields::geom_array_type InternalForce() {
  return safeFieldLookup<typename Fields::geom_array_type>("internal force");
}

template <class Fields>
typename Fields::geom_array_type NodalIndicator() {
  return safeFieldLookup<typename Fields::geom_array_type>("nodal indicator");
}

template <class Fields>
typename Fields::array_type NodalMass() {
  return safeFieldLookup<typename Fields::array_type>("nodal mass");
}

template <class Fields>
typename Fields::array_type NodalVolume() {
  return safeFieldLookup<typename Fields::array_type>("nodal volume");
}

template <class Fields>
typename Fields::array_type NodalPressure() {
  return safeFieldLookup<typename Fields::array_type>("nodal pressure");
}

template <class Fields>
typename Fields::array_type NodalPressureIncrement() {
  return safeFieldLookup<typename Fields::array_type>(
      "nodal pressure increment");
}

template <class Fields>
typename Fields::array_type ElementVolume() {
  return safeFieldLookup<typename Fields::array_type>("elem volume");
}

template <class Fields>
typename Fields::array_type InternalEnergyDensity() {
  return safeFieldLookup<typename Fields::array_type>(
      "internal energy density");
}

template <class Fields>
typename Fields::array_type ElementMass() {
  return safeFieldLookup<typename Fields::array_type>("elem mass");
}

template <class Fields>
typename Fields::array_type ElementInternalEnergy() {
  return safeFieldLookup<typename Fields::array_type>(
      "element internal energy");
}

template <class Fields>
typename Fields::array_type ElementJouleEnergy() {
  return safeFieldLookup<typename Fields::array_type>(
      "element joule energy");
}

template <class Fields>
typename Fields::array_type ElementTimeStep() {
  return safeFieldLookup<typename Fields::array_type>("element time step");
}

template <class Fields>
typename Fields::array_type FineScalePressure() {
  return safeFieldLookup<typename Fields::array_type>("fine scale pressure");
}

template <class Fields>
typename Fields::state_array_type MassDensity() {
  return safeFieldLookup<typename Fields::state_array_type>(
      "spatial deformed density");
}

template <class Fields>
typename Fields::array_type UserMatID() {
  return safeFieldLookup<typename Fields::array_type>("user mat id");
}

template <class Fields>
typename Fields::state_array_type InternalEnergyPerUnitMass() {
  return safeFieldLookup<typename Fields::state_array_type>(
      "internal energy per unit mass");
}

template <class Fields>
typename Fields::state_array_type PlaneWaveModulus() {
  return safeFieldLookup<typename Fields::state_array_type>(
      "plane wave modulus");
}

template <class Fields>
typename Fields::state_array_type BulkModulus() {
  return safeFieldLookup<typename Fields::state_array_type>("bulk modulus");
}

template <class Fields>
typename Fields::elem_vector_state_type FineScaleDisplacement() {
  return safeFieldLookup<typename Fields::elem_vector_state_type>(
      "fine scale displacement");
}

template <class Fields>
typename Fields::elem_vector_type FineScaleVelocity() {
  return safeFieldLookup<typename Fields::elem_vector_type>(
      "fine scale velocity");
}

template <class Fields>
typename Fields::elem_vector_type ElementShockHeatFlux() {
  return safeFieldLookup<typename Fields::elem_vector_type>(
      "element shock heat flux");
}

template <class Fields>
typename Fields::elem_sym_tensor_state_type Stress() {
  return safeFieldLookup<typename Fields::elem_sym_tensor_state_type>("stress");
}

template <class Fields>
typename Fields::elem_node_geom_type ElementForce() {
  return safeFieldLookup<typename Fields::elem_node_geom_type>("element force");
}

template <class Fields>
typename Fields::elem_tensor_type VelocityGradient() {
  return safeFieldLookup<typename Fields::elem_tensor_type>(
      "velocity gradient");
}

template <class Fields>
typename Fields::elem_tensor_type DeformationGradient() {
  return safeFieldLookup<typename Fields::elem_tensor_type>(
      "deformation gradient");
}

template <class Fields>
typename Fields::array_type Conductivity() {
  return safeFieldLookup<typename Fields::array_type>("conductivity");
}

template <class Fields>
typename Fields::array_type ElectricPotential() {
  return safeFieldLookup<typename Fields::array_type>("potential");
}

template <class Fields>
typename Fields::array_type MagneticFaceFlux() {
  return safeFieldLookup<typename Fields::array_type>(
      "magnetic face flux");
}

extern template void FieldDB_Finalize<1>();
extern template void FieldDB_Finalize<2>();
extern template void FieldDB_Finalize<3>();

} /* namespace lgr */

#endif
