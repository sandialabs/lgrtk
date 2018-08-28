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

#include "MeshFixture.hpp"
#include <cassert>

namespace lgr {

template <int SpatialDim>
void MeshFixture<SpatialDim>::verify(
    const typename FEMeshType::node_coords_type::HostMirror &  node_coords,
    const typename FEMeshType::elem_node_ids_type::HostMirror &elem_node_ids,
    const typename FEMeshType::node_elem_ids_type::HostMirror
        &node_elem_ids) {
  typedef typename FEMeshType::size_type size_type;

  const size_type node_count_total = node_coords.extent(0);
  const size_type elem_count_total = elem_node_ids.extent(0);

  for (size_type node_index = 0; node_index < node_count_total;
       ++node_index) {
    for (size_type j = node_elem_ids.row_map[node_index];
         j < node_elem_ids.row_map[node_index + 1]; ++j) {
      const size_type elem_index = node_elem_ids.entries(j, 0);
      const size_type node_local = node_elem_ids.entries(j, 1);
      const size_type en_id = elem_node_ids(elem_index, node_local);

      if (node_index != en_id) {
        std::ostringstream msg;
        msg << "MeshFixture node_elem_ids error"
            << " : node_index(" << node_index << ") entry(" << j
            << ") elem_index(" << elem_index << ") node_local(" << node_local
            << ") elem_node_id(" << en_id << ")";
        throw std::runtime_error(msg.str());
      }
    }
  }
}

template <int SpatialDim>
typename MeshFixture<SpatialDim>::FEMeshType
MeshFixture<SpatialDim>::create(MeshIO &meshio, comm::Machine machine) {
  // Finite element mesh:

  FEMeshType mesh;

  mesh.omega_h_mesh = meshio.getMesh();

  if(mesh.numDim != mesh.omega_h_mesh->dim() )
    throw std::runtime_error("FEMesh spatial dim template parameter must match Omega_h spatial dimension.");

  mesh.machine = machine;

  mesh.resetSizes();

  // Element counts:

  const size_t elem_count = mesh.omega_h_mesh->nelems();
  const size_t face_count = mesh.omega_h_mesh->nfaces();

  //! list of total nodes on the rank
  size_t node_count = mesh.omega_h_mesh->nverts();

  // Allocate the initial arrays. Note that the mesh.reAlloc() function does the resize
  if (node_count) {
    mesh.node_coords = node_coords_type("node_coords", mesh.geom_layout);
  }

  if (elem_count) {
    mesh.elem_node_ids = elem_node_ids_type("elem_node_ids", elem_count);
    mesh.elem_face_ids = elem_face_ids_type("elem_face_ids", elem_count);
    mesh.elem_face_orientations = elem_face_orient_type("elem_face_orientations", elem_count);
  }
  if (face_count) {
    mesh.face_node_ids = face_node_ids_type("face_node_ids", face_count);
  }

  mesh.updateMesh();

  return mesh;
}

template class MeshFixture<3>;
template class MeshFixture<2>;
template class MeshFixture<1>;

}  // end namespace lgr
