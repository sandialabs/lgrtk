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

#ifndef LGR_MESH_FIXTURE_HPP
#define LGR_MESH_FIXTURE_HPP

#include "MeshIO.hpp"
#include "LGR_Types.hpp"
#include "FEMesh.hpp"

namespace lgr {

template <int SpatialDim>
struct MeshFixture {
  typedef Scalar coordinate_scalar_type;
  typedef ExecSpace execution_space;

  static const unsigned element_node_count = SpatialDim + 1;

  typedef lgr::FEMesh<SpatialDim> FEMeshType;

  typedef typename FEMeshType::node_coords_type   node_coords_type;
  typedef typename FEMeshType::elem_node_ids_type elem_node_ids_type;
  typedef typename FEMeshType::face_node_ids_type face_node_ids_type;
  typedef typename FEMeshType::elem_face_ids_type elem_face_ids_type;
  typedef typename FEMeshType::elem_face_orient_type elem_face_orient_type;
  typedef typename FEMeshType::node_elem_ids_type node_elem_ids_type;

  static void verify(
      const typename FEMeshType::node_coords_type::HostMirror &  node_coords,
      const typename FEMeshType::elem_node_ids_type::HostMirror &elem_node_ids,
      const typename FEMeshType::node_elem_ids_type::HostMirror
          &node_elem_ids);

  static FEMeshType create(MeshIO &meshio, comm::Machine machine);
};

}  // end namespace lgr

#endif
