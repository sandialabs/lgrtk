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

#include "FEMesh.hpp"
#include "LGRLambda.hpp"

#include <Omega_h_mesh.hpp>
#include <Omega_h_align.hpp>

namespace lgr {

template <int SpatialDim>
void FEMesh<SpatialDim>::reportTags() const {
  // prints tags available to console, for error reports when things aren't found
  std::cout << "Omega_h mesh has the following tags:\n";
  for (int n = 0; n < 4; n++) {
    int tagCount = omega_h_mesh->ntags(n);
    std::cout << "  d = " << n << ": " << tagCount << " tags:\n";
    for (int tagOrdinal = 0; tagOrdinal < tagCount; tagOrdinal++) {
      std::cout << "    " << omega_h_mesh->get_tag(n, tagOrdinal)->name()
                << std::endl;
    }
  }
}

template <int SpatialDim>
Kokkos::View<const Omega_h::Real*> FEMesh<SpatialDim>::getFieldView(
    int dim, std::string const& name) const {
  if (omega_h_mesh->has_tag(dim, name)) {
    return omega_h_mesh->get_array<Omega_h::Real>(dim, name).view();
  } else {
    std::cout << "Error: Omega_h mesh does not have a tag with dim " << dim
              << " named \"" << name << "\"\n";
    reportTags();
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "tag not found");
  }
}

template <int SpatialDim>
void FEMesh<SpatialDim>::addFieldView(
    int                          dim,
    std::string const&           name,
    int                          ncomps,
    Kokkos::View<Omega_h::Real*> view) const {
  omega_h_mesh->add_tag(
      dim, name, ncomps, Omega_h::Reals(Omega_h::Write<Omega_h::Real>(view)));
}

template <int SpatialDim>
void FEMesh<SpatialDim>::setFieldView(
    int dim, std::string const& name, Kokkos::View<Omega_h::Real*> view) const {
  omega_h_mesh->set_tag(
      dim, name, Omega_h::Reals(Omega_h::Write<Omega_h::Real>(view)));
}

template <int SpatialDim>
void FEMesh<SpatialDim>::copyGeomFromMesh(
    int dim, std::string const& name, geom_array_type a) {
  auto nents = omega_h_mesh->nents(dim);
  auto spaceDim = omega_h_mesh->dim();
  assert(static_cast<decltype(spaceDim)>(a.size()) == nents * spaceDim);
  auto in = getFieldView(dim, name);
  auto f = LAMBDA_EXPRESSION(Omega_h::LO ent) {
    for (Omega_h::Int j = 0; j < spaceDim; ++j) {
      a(ent, j) = in(ent * spaceDim + j);
    }
  };
  Kokkos::parallel_for(nents, f);
}

template <int SpatialDim>
void FEMesh<SpatialDim>::resetSizes() {
  nelems = omega_h_mesh->nelems();
  nnodes = omega_h_mesh->nverts();
  nfaces = Omega_h::FACE <= omega_h_mesh->dim() ? omega_h_mesh->nfaces() : 0;
  auto      spaceDim = omega_h_mesh->dim();
  const int dimensions[] = {static_cast<int>(nnodes), spaceDim};
  const int order[] = {1, 0};
  const int rank = 2;
  geom_layout = Kokkos::LayoutStride::order_dimensions(rank, order, dimensions);
}

template <int SpatialDim>
void FEMesh<SpatialDim>::reAlloc() {
  node_coords = node_coords_type("node_coords", geom_layout);
  Kokkos::realloc(elem_node_ids, nelems);
  Kokkos::realloc(face_node_ids, nfaces);
  Kokkos::realloc(elem_face_ids, nelems);
  Kokkos::realloc(elem_face_orientations, nelems);
}

template <int SpatialDim>
void FEMesh<SpatialDim>::updateMesh() {
  copyGeomFromMesh(0, "coordinates", node_coords);
  auto ev2v = omega_h_mesh->ask_verts_of(SpatialDim);
  auto fv2v = omega_h_mesh->ask_verts_of(SpatialDim-1);

  // as of CUDA version 7.5.7, lambdas have trouble
  // capturing class members; captured variables really
  // need to be in local scope.
  elem_node_ids_type local_elem_node_ids = elem_node_ids;
  auto               f0 = LAMBDA_EXPRESSION(size_t i) {
    for (size_t j = 0; j < ElemNodeCount; ++j)
      local_elem_node_ids(i, j) = ev2v[i * ElemNodeCount + j];
  };
  Kokkos::parallel_for(nelems, f0);

  face_node_ids_type local_face_node_ids = face_node_ids;
  auto               e0 = LAMBDA_EXPRESSION(size_t i) {
    for (size_t j = 0; j < FaceNodeCount; ++j)
      local_face_node_ids(i, j) = fv2v[i * FaceNodeCount + j];
  };
  Kokkos::parallel_for(nfaces, e0);
  
  if (SpatialDim>2) { 
    auto ev2f = omega_h_mesh->ask_down(SpatialDim,Omega_h::FACE).ab2b;
    auto ev2o = omega_h_mesh->ask_down(SpatialDim,Omega_h::FACE).codes;
    elem_face_ids_type local_elem_face_ids = elem_face_ids;
    elem_face_orient_type local_elem_face_orientations = elem_face_orientations;
    auto getFaces = LAMBDA_EXPRESSION(size_t elemIndex) {
      for (size_t f = 0; f<ElemFaceCount; ++f) {
	local_elem_face_ids(elemIndex, f) = ev2f[elemIndex*ElemFaceCount + f];
	const bool flipped = Omega_h::code_is_flipped(ev2o[elemIndex*ElemFaceCount + f]);
	const int sign = flipped ? -1 : +1;
	local_elem_face_orientations(elemIndex, f) = sign;
      }
    };
    Kokkos::parallel_for(nelems, getFaces);
  }

  //------------------------------------
  // Populate node->element connectivity:

  auto v2e = omega_h_mesh->ask_up(0, SpatialDim);
  // row_map_type is defined with constant entries, we want to build
  // one by hand:
  //typename node_elem_ids_type::row_map_type
  Kokkos::View<
      typename node_elem_ids_type::size_type*,
      typename node_elem_ids_type::array_layout,
      typename node_elem_ids_type::device_type>
      row_map("node elem row map", nnodes + 1);

  auto f1 = LAMBDA_EXPRESSION(size_t i) { row_map[i] = v2e.a2ab[i]; };
  Kokkos::parallel_for(nnodes + 1, f1);

  typename node_elem_ids_type::entries_type entries(
      "node elem entries", nelems * ElemNodeCount);

  auto f2 = LAMBDA_EXPRESSION(size_t i) {
    entries(i, 0) = v2e.ab2b[i];
    entries(i, 1) = Omega_h::code_which_down(v2e.codes[i]);
  };
  Kokkos::parallel_for(nelems * ElemNodeCount, f2);

  node_elem_ids = node_elem_ids_type(entries, row_map);
}

template struct FEMesh<1>;
template struct FEMesh<2>;
template struct FEMesh<3>;

} /* namespace lgr */
