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

#ifndef LGR_FEMESH_HPP
#define LGR_FEMESH_HPP

#include "LGR_Types.hpp"
#include "ParallelComm.hpp"
#include <Kokkos_StaticCrsGraph.hpp>
#include <Omega_h_defines.hpp>

namespace Omega_h {
class Mesh;
}

namespace lgr {

template <int SpatialDim>
struct FEMesh {

  // lowest-order simplices always have 1 more than SpatialDim nodes
  static constexpr int ElemNodeCount = SpatialDim + 1;
  static constexpr int FaceNodeCount = SpatialDim;
  static constexpr int ElemFaceCount = SpatialDim + 1;

  typedef typename ExecSpace::size_type size_type;
  typedef ExecSpace                     execution_space;
  typedef unsigned                   local_ordinal_type;

  static constexpr size_type element_node_count = ElemNodeCount;

  // lgr vector data (note the Kokkos::LayoutStride)
  typedef Kokkos::
      View<Scalar * [SpatialDim], Kokkos::LayoutStride, execution_space>
          geom_array_type;

  typedef geom_array_type                                   node_coords_type;
  typedef Kokkos::View<size_type * [ElemNodeCount], ExecSpace> elem_node_ids_type;
  typedef Kokkos::View<size_type * [FaceNodeCount], ExecSpace> face_node_ids_type;
  typedef Kokkos::View<size_type * [ElemFaceCount], ExecSpace> elem_face_ids_type;
  typedef Kokkos::View<int * [ElemFaceCount], ExecSpace> elem_face_orient_type;
  typedef Kokkos::StaticCrsGraph<size_type[2], ExecSpace>      node_elem_ids_type;
  
  node_coords_type   node_coords;
  elem_node_ids_type elem_node_ids;
  face_node_ids_type face_node_ids;
  elem_face_ids_type elem_face_ids;
  elem_face_orient_type elem_face_orientations;
  node_elem_ids_type node_elem_ids;

  comm::Machine machine;

  size_t nelems;
  size_t nnodes;
  size_t nfaces;

  Omega_h::Mesh* omega_h_mesh;
  static constexpr int  numDim = SpatialDim;

  Kokkos::LayoutStride geom_layout;

  void resetSizes();
  void reAlloc();
  void reportTags() const;
  void updateMesh();

  int getNumRanks() { return comm::size(machine); }

  Kokkos::View<const Omega_h::Real*> getFieldView(
      int dim, std::string const& name) const;
  void addFieldView(
      int                          dim,
      std::string const&           name,
      int                          ncomps,
      Kokkos::View<Omega_h::Real*> view) const;
  void setFieldView(
      int                          dim,
      std::string const&           name,
      Kokkos::View<Omega_h::Real*> view) const;
  void copyGeomFromMesh(int dim, std::string const& name, geom_array_type a);
};

} /* namespace lgr */

#endif
