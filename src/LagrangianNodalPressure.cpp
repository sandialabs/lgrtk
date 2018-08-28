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

#include "LagrangianNodalPressure.hpp"
#include "FieldDB.hpp"
#include "LGRLambda.hpp"
#include "ElementHelpers.hpp"
#include "TensorOperations_inline.hpp"
namespace lgr {

template <int SpatialDim>
AssembleNodalPressureEquation<SpatialDim>::AssembleNodalPressureEquation(
      const Fields &mesh_fields, int arg_state0, int arg_state1)
      : elem_node_connectivity(mesh_fields.femesh.elem_node_ids)
      , stress(Stress<Fields>())
      , nodal_volume(NodalVolume<Fields>())
      , nodal_pressure(NodalPressure<Fields>())
      , nodal_pressure_increment(NodalPressureIncrement<Fields>())
      , updatedCoordinates(Coordinates<Fields>())
      , bulkModulus(BulkModulus<Fields>())
      , uprime(FineScaleDisplacement<Fields>())
      , volume_contribution("volume_contribution", mesh_fields.femesh.nelems)
      , pressure_contribution(
            "pressure_contribution", mesh_fields.femesh.nelems)
      , pressure_increment_contribution(
            "pressure_increment_contribution", mesh_fields.femesh.nelems)
      , node_elem_ids(mesh_fields.femesh.node_elem_ids)
      , state0(arg_state0)
      , state1(arg_state1) {}

template <int SpatialDim>
KOKKOS_INLINE_FUNCTION
void AssembleNodalPressureEquation<SpatialDim>::operator()(ElemLoopTag, int ielem) const {
  Scalar x[ElemNodeCount], y[ElemNodeCount], z[ElemNodeCount];
  Scalar grad_x[ElemNodeCount], grad_y[ElemNodeCount], grad_z[ElemNodeCount];
  typename Fields::geom_array_type oc(
      Fields::getGeomFromSA(updatedCoordinates, state0));
  typename Fields::geom_array_type nc(
      Fields::getGeomFromSA(updatedCoordinates, state1));
  for (int i = 0; i < ElemNodeCount; ++i) {
    const int n = elem_node_connectivity(ielem, i);
    x[i] = 0.5 * (oc(n, 0) + nc(n, 0));
    y[i] = 0.5 * (oc(n, 1) + nc(n, 1));
    z[i] = 0.5 * (oc(n, 2) + nc(n, 2));
  }
  comp_grad(x, y, z, grad_x, grad_y, grad_z);
  Scalar volume = dot4(x, grad_x);

  //pressure projection terms
  Scalar pressure = 0;

  static_assert(SpatialDim == 3 || SpatialDim == 2, "SpatialDim template parameter must be 2 or 3");

  if(SpatialDim == 3) {
      pressure -= 0.5*( stress(ielem, FieldsEnum<3>::K_S_XX, state0) + stress(ielem, FieldsEnum<3>::K_S_XX, state1) );
      pressure -= 0.5*( stress(ielem, FieldsEnum<3>::K_S_YY, state0) + stress(ielem, FieldsEnum<3>::K_S_YY, state1) );
      pressure -= 0.5*( stress(ielem, FieldsEnum<3>::K_S_ZZ, state0) + stress(ielem, FieldsEnum<3>::K_S_ZZ, state1) );
  } else {
      pressure -= 0.5*( stress(ielem, FieldsEnum<2>::K_S_XX, state0) + stress(ielem, FieldsEnum<2>::K_S_XX, state1) );
      pressure -= 0.5*( stress(ielem, FieldsEnum<2>::K_S_YY, state0) + stress(ielem, FieldsEnum<3>::K_S_YY, state1) );
  }

  pressure /= 3.0;
  Scalar deltaPressure(0);
  if(SpatialDim == 3) {
      deltaPressure -= ( stress(ielem, FieldsEnum<3>::K_S_XX, state1) - stress(ielem, FieldsEnum<3>::K_S_XX, state0) );
      deltaPressure -= ( stress(ielem, FieldsEnum<3>::K_S_YY, state1) - stress(ielem, FieldsEnum<3>::K_S_YY, state0) );
      deltaPressure -= ( stress(ielem, FieldsEnum<3>::K_S_ZZ, state1) - stress(ielem, FieldsEnum<3>::K_S_ZZ, state0) );
  } else {
      deltaPressure -= ( stress(ielem, FieldsEnum<2>::K_S_XX, state1) - stress(ielem, FieldsEnum<2>::K_S_XX, state0) );
      deltaPressure -= ( stress(ielem, FieldsEnum<2>::K_S_YY, state1) - stress(ielem, FieldsEnum<3>::K_S_YY, state0) );
  }

  deltaPressure /= 3.0;

  pressure *= volume;
  deltaPressure *= volume;

  volume /= ElemNodeCount;
  pressure /= ElemNodeCount;
  deltaPressure /= ElemNodeCount;

  for (int i = 0; i < Fields::ElemNodeCount; ++i) {
    volume_contribution(ielem, i) += volume;
    pressure_contribution(ielem, i) += pressure;
    pressure_increment_contribution(ielem, i) += deltaPressure;
  }

  //uprime terms
  Scalar el_uprime[] = {0., 0., 0.};
  Scalar el_deltaUprime[] = {0., 0., 0.};
  for (int slot = 0; slot < Fields::SpaceDim; ++slot) {
    el_uprime[slot] =
        0.5 * (uprime(ielem, slot, state0) + uprime(ielem, slot, state1));
    el_deltaUprime[slot] =
        (uprime(ielem, slot, state1) - uprime(ielem, slot, state0));
  }

  for (int i = 0; i < ElemNodeCount; ++i) {
    Scalar gradInnerUprime = grad_x[i] * el_uprime[0] +
                             grad_y[i] * el_uprime[1] +
                             grad_z[i] * el_uprime[2];
    gradInnerUprime *= (-1);
    pressure_contribution(ielem, i) -= gradInnerUprime;

    Scalar gradInnerDeltaUprime = grad_x[i] * el_deltaUprime[0] +
                                  grad_y[i] * el_deltaUprime[1] +
                                  grad_z[i] * el_deltaUprime[2];
    gradInnerDeltaUprime *= (-1);
    pressure_increment_contribution(ielem, i) -= gradInnerDeltaUprime;
  }
}

template <int SpaceDim>
KOKKOS_INLINE_FUNCTION
void AssembleNodalPressureEquation<SpaceDim>::operator()(NodeLoopTag, int inode) const {
  auto begin = node_elem_ids.row_map(inode);
  auto end = node_elem_ids.row_map(inode + 1);
  for (auto i = begin; i < end; ++i) {
    auto ielem = node_elem_ids.entries(i, 0);
    auto which_down = node_elem_ids.entries(i, 1);
    nodal_volume(inode) += volume_contribution(ielem, which_down);
    nodal_pressure(inode) += pressure_contribution(ielem, which_down);
    nodal_pressure_increment(inode) +=
        pressure_increment_contribution(ielem, which_down);
  }
}

template <int SpaceDim>
void AssembleNodalPressureEquation<SpaceDim>::apply(const Fields &mesh_fields) {
  Kokkos::parallel_for(
      Kokkos::RangePolicy<ExecSpace, ElemLoopTag>(
          0, mesh_fields.femesh.nelems),
      *this);
  Kokkos::parallel_for(
      Kokkos::RangePolicy<ExecSpace, NodeLoopTag>(
          0, mesh_fields.femesh.nnodes),
      *this);
}

template <int SpatialDim>
LagrangianNodalPressure<SpatialDim>::LagrangianNodalPressure(Fields &mesh_fields, int arg_state0, int arg_state1)
      : meshFields_(mesh_fields), state0_(arg_state0), state1_(arg_state1) {}

template <int SpatialDim>
void LagrangianNodalPressure<SpatialDim>::zeroData() {
  //initialize nodal volume and pressure fields
  const typename Fields::array_type &nodal_volume = NodalVolume<Fields>();
  const typename Fields::array_type &nodal_pressure = NodalPressure<Fields>();
  const typename Fields::array_type &nodal_pressure_increment =
      NodalPressureIncrement<Fields>();
  Kokkos::deep_copy(nodal_volume, 0.0);
  Kokkos::deep_copy(nodal_pressure, 0.0);
  Kokkos::deep_copy(nodal_pressure_increment, 0.0);
}

template <int SpatialDim>
void LagrangianNodalPressure<SpatialDim>::computeNodalPressure() {
  this->zeroData();

  AssembleNodalPressureEquation<SpatialDim> anpe(
      meshFields_, state0_, state1_);
  anpe.apply(meshFields_);

  meshFields_.conform("nodal_volume", NodalVolume<Fields>());
  meshFields_.conform("nodal_pressure", NodalPressure<Fields>());
  meshFields_.conform(
      "nodal_pressure_increment", NodalPressureIncrement<Fields>());

  //compute nodal presssure (solve pressure equation)
  {
    auto nodal_volume = NodalVolume<Fields>();
    auto nodal_pressure = NodalPressure<Fields>();
    auto nodal_pressure_increment = NodalPressureIncrement<Fields>();
    auto computePressure = LAMBDA_EXPRESSION(int inode) {
      const Scalar v = nodal_volume(inode);
      Scalar &     p = nodal_pressure(inode);
      Scalar &     deltaP = nodal_pressure_increment(inode);
      p /= v;
      deltaP /= v;
    };  //end lambda computePressure
    Kokkos::parallel_for(meshFields_.femesh.nnodes, computePressure);
  }
}

#define LGR_EXPL_INST(SpatialDim) \
template struct AssembleNodalPressureEquation<SpatialDim>; \
template class LagrangianNodalPressure<SpatialDim>;
LGR_EXPL_INST(3)
LGR_EXPL_INST(2)
#undef LGR_EXPL_INST

}  //end namespace lgr
