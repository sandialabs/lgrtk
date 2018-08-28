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

#ifndef LGR_LAGRANGIAN_FINE_SCALE_HPP
#define LGR_LAGRANGIAN_FINE_SCALE_HPP

#include "Fields.hpp"
#include "MeshFixture.hpp"

namespace lgr {

template <int SpatialDim>
class LagrangianFineScale {
 public:
  using FixtureType = MeshFixture<SpatialDim>;
  typedef typename FixtureType::execution_space              execution_space;
  typedef lgr::Fields<SpatialDim> Fields;
  static const int ElemNodeCount = Fields::ElemNodeCount;

 private:
  typename Fields::elem_node_ids_type           elem_node_connectivity;
  typename Fields::geom_state_array_type        updatedCoordinates;
  const typename Fields::array_type             nodal_volume;
  const typename Fields::array_type             nodal_mass;
  const typename Fields::array_type             elem_mass;
  const typename Fields::array_type             pprime;
  const typename Fields::elem_vector_state_type uprime;
  const typename Fields::elem_vector_type       vprime;
  const typename Fields::state_array_type       bulkModulus;
  const typename Fields::state_array_type       planeWaveModulus;
  const typename Fields::array_type             nodal_pressure;
  const typename Fields::array_type             nodal_pressure_increment;
  int                                           state0_;
  int                                           state1_;
  Scalar                                        c_tau_;
  int                                           nelems_;
  Scalar                                        dt_;
  typename Fields::geom_array_type              velocity[2];

 public:
  LagrangianFineScale(
      Fields &mesh_fields, int state0_in, int state1_in, Scalar c_tau_in);

  KOKKOS_INLINE_FUNCTION
  void operator()(int ielem) const;

  void apply(Scalar timeStep);

};

extern template class LagrangianFineScale<3>;
extern template class LagrangianFineScale<2>;

}  //end namespace lgr

#endif
