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

#ifndef LGR_FIELDS_HPP
#define LGR_FIELDS_HPP

#include "LGR_Types.hpp"
#include "FEMesh.hpp"
#include <Teuchos_ParameterList.hpp>
#include <Omega_h_mesh.hpp>

namespace lgr {

template <int SpatialDim>
struct Fields {
  static const int NumStates = 2;

  typedef Scalar                              scalar_type;
  typedef ExecSpace                              execution_space;
  typedef typename execution_space::size_type size_type;

  typedef lgr::FEMesh<SpatialDim> FEMesh;
  static int constexpr ElemNodeCount = FEMesh::ElemNodeCount;
  static int constexpr FaceNodeCount = FEMesh::FaceNodeCount;
  static int constexpr ElemFaceCount = FEMesh::ElemFaceCount;
  static int constexpr SymTensorLength = (SpatialDim * SpatialDim + SpatialDim) / 2;
  static int constexpr TensorLength = SpatialDim * SpatialDim;
  static int constexpr SpaceDim = SpatialDim;

  typedef typename FEMesh::local_ordinal_type local_ordinal_type;
  typedef typename FEMesh::node_coords_type   node_coords_type;
  typedef typename FEMesh::elem_node_ids_type elem_node_ids_type;
  typedef typename FEMesh::face_node_ids_type face_node_ids_type;
  typedef typename FEMesh::elem_face_ids_type elem_face_ids_type;
  typedef typename FEMesh::elem_face_orient_type elem_face_orient_type;
  typedef typename FEMesh::node_elem_ids_type node_elem_ids_type;
  typedef typename FEMesh::geom_array_type    geom_array_type;

  typedef Kokkos::View<Scalar*, execution_space> array_type;
  typedef Kokkos::
      View<Scalar * [NumStates], Kokkos::LayoutLeft, execution_space>
          state_array_type;

  typedef Kokkos::View<
      Scalar * [SpatialDim][NumStates],
      Kokkos::LayoutStride,
      execution_space>
      geom_state_array_type;

  // Accessor function for individual states
  static KOKKOS_INLINE_FUNCTION geom_array_type
                                getGeomFromSA(const geom_state_array_type& a, const int state) {
    return Kokkos::subview(a, Kokkos::ALL(), Kokkos::ALL(), state);
  }
  static KOKKOS_INLINE_FUNCTION array_type
                                getFromSA(const state_array_type& a, const int state) {
    return Kokkos::subview(a, Kokkos::ALL(), state);
  }

  //element data (default stride)
  typedef Kokkos::View<Scalar * [SpatialDim], execution_space> elem_vector_type;
  typedef Kokkos::View<Scalar * [SymTensorLength], 
      Kokkos::LayoutStride,
      execution_space>
      elem_sym_tensor_type;

  typedef Kokkos::View<Scalar * [TensorLength], execution_space>
      elem_tensor_type;

  typedef Kokkos::View<Scalar * [SpatialDim][NumStates], execution_space>
      elem_vector_state_type;
  typedef Kokkos::View<Scalar * [SymTensorLength][NumStates], execution_space>
      elem_sym_tensor_state_type;
  typedef Kokkos::View<Scalar * [TensorLength][NumStates], execution_space>
      elem_tensor_state_type;
  typedef Kokkos::View<Scalar * [SpatialDim][ElemNodeCount], execution_space>
                                              elem_node_geom_type;
  typedef Kokkos::View<int*, execution_space> index_array_type;

  static KOKKOS_INLINE_FUNCTION elem_sym_tensor_type
                                getFromSymTensorSA(const elem_sym_tensor_state_type& a, const int state) {
    return Kokkos::subview(a, Kokkos::ALL(), Kokkos::ALL(), state);
  }


  // Mesh:
  FEMesh femesh;

  Teuchos::ParameterList& fieldData;

  Fields(const FEMesh& mesh, Teuchos::ParameterList& data);

  // Resize all fields as the mesh has changed
  void resize();
  void allocate_and_resize_fields();

  // Data copying to and from Omega_h
  void copyGeomToMesh(
      int                dim,
      std::string const& name,
      geom_array_type    from,
      bool               should_add = true) const;
  void copyGeomToMesh(std::string const& name, geom_array_type from) const;
  void copyGeomFromMesh(int dim, std::string const& name, geom_array_type into);
  void copyGeomFromMesh(std::string const& name, geom_array_type into);
  void copyElemScalarFromMesh(char const* name, array_type into);
  void copyToMesh(char const* name, const array_type from) const;
  void copyFromMesh(char const* name, array_type into);
  void copyElemTensorFromMesh(char const* name, elem_tensor_type into);
  void copyElemScalarToMesh(char const* name, const array_type from) const;
  void copyElemTensorToMesh(
      char const* name, const elem_tensor_type from) const;
  void copyElemSymTensorToMesh(
      char const* name, const elem_sym_tensor_type from) const;
  // Parallel field synchronization via Omega_h
  void conformGeom(char const* name, geom_array_type a);
  void conform(char const* name, array_type a);

  void copyTagsFromMesh(
      Omega_h::TagSet const& tags,
      int                    state,
      int                    end_state);  // for multi-state variables, copies up to but not including end_state
  void copyTagsToMesh(Omega_h::TagSet const& tags, int state) const;
  void copyTagsFromMesh(Omega_h::TagSet const& tags, int state);
  void cleanTagsFromMesh(Omega_h::TagSet const& tags) const;
  void copyCoordsToMesh(int state) const;
  void copyCoordsFromMesh(int state);
};

extern template struct Fields<1>;
extern template struct Fields<2>;
extern template struct Fields<3>;

} /* namespace lgr */

#endif
