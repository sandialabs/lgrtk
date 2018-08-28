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

#ifndef LGR_INITIAL_CONDITIONS_HPP
#define LGR_INITIAL_CONDITIONS_HPP

#include <memory>
#include <vector>
#include <string>

#include "Fields.hpp"

#include <Omega_h_assoc.hpp>

namespace Teuchos {
class ParameterList;
}

namespace lgr {

template <class Field>
class InitialCondition {
  explicit InitialCondition(const InitialCondition &);
  InitialCondition &operator=(const InitialCondition &);
  InitialCondition();

 protected:
  const std::string        name;
  std::vector<std::string> element_blocks;
  std::vector<std::string> node_sets;
  std::vector<std::string> face_sets;

 public:
  typedef typename Field::geom_array_type    geom_array_type;
  typedef typename Field::array_type         array_type;
  typedef typename Field::node_coords_type   node_coords;
  typedef typename Field::elem_node_ids_type elem_node_ids;
  typedef typename Field::face_node_ids_type face_node_ids;
  typedef typename Field::elem_face_ids_type elem_face_ids;
  InitialCondition(const std::string &n, Teuchos::ParameterList &params);
  virtual void set(
      const Omega_h::MeshDimSets &,
      const geom_array_type,
      const node_coords) = 0;
  virtual void set(
      const Omega_h::MeshDimSets &,
      const array_type,
      const node_coords,
      const elem_node_ids) = 0;
  virtual void set(
      const Omega_h::MeshDimSets &,
      const Omega_h::MeshDimSets &,
      const array_type,
      const node_coords,
      const face_node_ids,
      const elem_face_ids) = 0;
  virtual ~InitialCondition();
};

template <class Field>
class ConstantInitialCondition : public InitialCondition<Field> {
 public:
  using Parent = InitialCondition<Field>;
  using typename Parent::array_type;
  using typename Parent::elem_node_ids;
  using typename Parent::face_node_ids;
  using typename Parent::elem_face_ids;
  using typename Parent::geom_array_type;
  using typename Parent::node_coords;
  ConstantInitialCondition(
      const std::string &name, Teuchos::ParameterList &params);
  void set(
      const Omega_h::MeshDimSets &,
      const geom_array_type,
      const node_coords) override final;
  void set(
      const Omega_h::MeshDimSets &,
      const array_type,
      const node_coords,
      const elem_node_ids) override final;
  void set(
      const Omega_h::MeshDimSets &,
      const Omega_h::MeshDimSets &,
      const array_type,
      const node_coords,
      const face_node_ids,                  
      const elem_face_ids) override final;

 private:
  std::vector<double> value;
};

template <class Field>
class FunctionInitialCondition : public InitialCondition<Field> {
  explicit FunctionInitialCondition(const FunctionInitialCondition &);
  FunctionInitialCondition &operator=(const FunctionInitialCondition &);

 public:
  using Parent = InitialCondition<Field>;
  using typename Parent::array_type;
  using typename Parent::elem_node_ids;
  using typename Parent::face_node_ids;
  using typename Parent::elem_face_ids;
  using typename Parent::geom_array_type;
  using typename Parent::node_coords;
  explicit FunctionInitialCondition(
      const std::string &name, Teuchos::ParameterList &params);
  void set(
      const Omega_h::MeshDimSets &,
      const geom_array_type,
      const node_coords) override final;
  void set(
      const Omega_h::MeshDimSets &,
      const array_type,
      const node_coords,
      const elem_node_ids) override final;
  void set(
      const Omega_h::MeshDimSets &,
      const Omega_h::MeshDimSets &,
      const array_type,
      const node_coords,
      const face_node_ids, 
      const elem_face_ids) override final;

 private:
  const std::string expr_string;
};

template <class Field>
class InitialConditions {
  explicit InitialConditions(const InitialConditions &);
  InitialConditions &operator=(const InitialConditions &);

 public:
  typedef typename Field::FEMesh FEMesh;

  typedef typename Field::geom_array_type       geom_array_type;
  typedef typename Field::geom_state_array_type geom_state_array_type;

  typedef typename Field::array_type       array_type;
  typedef typename Field::state_array_type state_array_type;

  explicit InitialConditions(Teuchos::ParameterList &params);

  void set(
      const Omega_h::MeshDimSets &nodesets,
      const geom_state_array_type velocity,
      const geom_array_type       dispacement,
      const FEMesh &              femesh);
  void set(
      const Omega_h::MeshDimSets &elementsets,
      const state_array_type      density,
      const state_array_type      internal_energy_per_unit_mass,
      const FEMesh &              femesh);
  void set(
      const Omega_h::MeshDimSets &facesets,
      const Omega_h::MeshDimSets &elementsets,
      const array_type            magnetic_face_flux,
      const FEMesh &              femesh);

 private:
  std::vector<std::shared_ptr<InitialCondition<Field>>>
      velocity_initial_conditions;
  std::vector<std::shared_ptr<InitialCondition<Field>>>
      displacement_initial_conditions;
  std::vector<std::shared_ptr<InitialCondition<Field>>>
      density_initial_conditions;
  std::vector<std::shared_ptr<InitialCondition<Field>>>
      internal_energy_initial_conditions;
  std::vector<std::shared_ptr<InitialCondition<Field>>>
      face_flux_initial_conditions;
};
#define LGR_EXPL_INST_DECL(SpatialDim) \
extern template class InitialCondition<Fields<SpatialDim>>; \
extern template class ConstantInitialCondition<Fields<SpatialDim>>; \
extern template class FunctionInitialCondition<Fields<SpatialDim>>; \
extern template class InitialConditions<Fields<SpatialDim>>;
LGR_EXPL_INST_DECL(3)
LGR_EXPL_INST_DECL(2)
#undef LGR_EXPL_INST_DECL

} /* namespace lgr */

#endif
