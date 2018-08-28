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

#ifndef LGR_NODAL_PRESSURE_HPP
#define LGR_NODAL_PRESSURE_HPP

#include "Fields.hpp"
#include "FieldsEnum.hpp"
#include "MeshFixture.hpp"

namespace lgr {

template <int SpatialDim>
struct AssembleNodalPressureEquation {
  struct PressureTag {};
  struct UprimeTag {};

  typedef lgr::Fields<SpatialDim> Fields;

  static const int ElemNodeCount = Fields::ElemNodeCount;

  const typename Fields::elem_node_ids_type         elem_node_connectivity;
  const typename Fields::elem_sym_tensor_state_type stress;
  const typename Fields::array_type                 nodal_volume;
  const typename Fields::array_type                 nodal_pressure;
  const typename Fields::array_type                 nodal_pressure_increment;
  const typename Fields::geom_state_array_type      updatedCoordinates;
  const typename Fields::state_array_type           bulkModulus;
  const typename Fields::elem_vector_state_type     uprime;
  /* temporaries for element -> node summation */
  using elem_node_scalar_type =
      Kokkos::View<Scalar * [ElemNodeCount], ExecSpace>;
  const elem_node_scalar_type               volume_contribution;
  const elem_node_scalar_type               pressure_contribution;
  const elem_node_scalar_type               pressure_increment_contribution;
  const typename Fields::node_elem_ids_type node_elem_ids;
  int                                       state0;
  int                                       state1;

  AssembleNodalPressureEquation(
      const Fields &mesh_fields, int arg_state0, int arg_state1);

  struct ElemLoopTag {};
  KOKKOS_INLINE_FUNCTION
  void operator()(ElemLoopTag, int ielem) const;

  struct NodeLoopTag {};
  KOKKOS_INLINE_FUNCTION
  void operator()(NodeLoopTag, int inode) const;

  void apply(const Fields &mesh_fields);

};  //end struct AssembleNodalPressureEquation

template <int SpatialDim>
class LagrangianNodalPressure {
 public:
  using FixtureType = MeshFixture<SpatialDim>;
  typedef typename FixtureType::execution_space              execution_space;
  typedef lgr::Fields<SpatialDim> Fields;

 private:
  Fields &meshFields_;
  int     state0_;
  int     state1_;

 public:
  LagrangianNodalPressure(Fields &mesh_fields, int arg_state0, int arg_state1);

  void zeroData();

  void computeNodalPressure();

};  //end class LagrangianNodalPressure

#define LGR_EXPL_INST_DECL(SpatialDim) \
extern template struct AssembleNodalPressureEquation<SpatialDim>; \
extern template class LagrangianNodalPressure<SpatialDim>;
LGR_EXPL_INST_DECL(3)
LGR_EXPL_INST_DECL(2)
#undef LGR_EXPL_INST_DECL

}  //end namespace lgr

#endif
