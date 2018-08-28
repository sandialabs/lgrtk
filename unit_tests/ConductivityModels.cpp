//@HEADER
// ************************************************************************
//
//                        LGR v. 1.0
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

#include "Teuchos_UnitTestHarness.hpp"

#include "Omega_h_build.hpp"
#include "Omega_h_map.hpp"
#include "Omega_h_matrix.hpp"
#include "LGRTestHelpers.hpp"

#include <ConductivityModels.hpp>
#include <FEMesh.hpp>
#include <Fields.hpp>
#include <FieldDB.hpp>


namespace {

TEUCHOS_UNIT_TEST(ConductivityModels, Constant)
{
  static const int SpatialDim = 3;
  typedef lgr::Fields<SpatialDim> Fields;
  const double tol = 1e-12;


  Teuchos::ParameterList ConductivityModelParameterList;
  Teuchos::ParameterList sublist;
  sublist.set("user id", 1);
  sublist.set("Model Type", "Constant");
  sublist.set("Element Block", "1");
  sublist.set("conductivity", 3.1415);
  ConductivityModelParameterList.set("some conductivity", sublist);

  Teuchos::ParameterList fieldData;

  Omega_h::MeshDimSets   elementSets;

  auto libOmegaH = lgr::getLibraryOmegaH();
  auto omega_h_mesh = build_box(libOmegaH->world(),OMEGA_H_SIMPLEX,1,1,1,4,4,4);
  lgr::FEMesh<SpatialDim> mesh;
  {
    mesh.omega_h_mesh = &omega_h_mesh;
    if(mesh.numDim != mesh.omega_h_mesh->dim() )
        throw std::runtime_error("FEMesh spatial dim template parameter must match Omega_h spatial dimension.");

    mesh.machine = lgr::getCommMachine();

    mesh.resetSizes();

    // Element counts:
    const size_t elem_count = mesh.omega_h_mesh->nelems();
    const size_t face_count = Omega_h::FACE <= mesh.omega_h_mesh->dim() ? mesh.omega_h_mesh->nfaces() : 0;

    // list of total nodes on the rank
    size_t node_count = mesh.omega_h_mesh->nverts();

    // Allocate the initial arrays. Note that the mesh.reAlloc() function does the resize
    if ( node_count ) {
      mesh.node_coords = typename Fields::node_coords_type( "node_coords", mesh.geom_layout );
    }
    if ( elem_count ) {
      mesh.elem_node_ids = typename Fields::elem_node_ids_type( "elem_node_ids", elem_count );
      mesh.elem_face_ids = typename Fields::elem_face_ids_type( "elem_face_ids", elem_count );
      mesh.elem_face_orientations = typename Fields::elem_face_orient_type( "elem_face_orientations", elem_count );
    }
    if ( face_count ) {
      mesh.face_node_ids = typename Fields::face_node_ids_type( "face_node_ids", face_count );
    }
    mesh.updateMesh();

    Omega_h::Read<Omega_h::I8> interiorMarks = Omega_h::mark_by_class_dim(mesh.omega_h_mesh, SpatialDim, SpatialDim);
    Omega_h::LOs               localOrdinals = Omega_h::collect_marked(interiorMarks);
    elementSets["1"] = localOrdinals;
  }

  Fields mesh_fields( mesh, fieldData );

  Kokkos::realloc(lgr::FieldDB<typename Fields::array_type>::Self()["conductivity"], mesh_fields.femesh.nelems);

  std::list<std::shared_ptr<lgr::ConductivityModelBase<SpatialDim>>> theConductivityModels;
  lgr::createConductivityModels(ConductivityModelParameterList, mesh_fields, elementSets, theConductivityModels);
  for ( auto condPtr : theConductivityModels ) condPtr->initializeElements(mesh_fields);
  for ( auto condPtr : theConductivityModels ) condPtr->updateElements    (mesh_fields, 0);
  for ( auto condPtr : theConductivityModels ) condPtr->updateElements    (mesh_fields, 1);
  Fields::array_type Conductivity = lgr::Conductivity<Fields>();

  const int numCells = Conductivity.extent(0);
  lgr::Scalar diff=0;
  Kokkos::parallel_reduce("TestConductivityField", numCells, LAMBDA_EXPRESSION(int cell, lgr::Scalar &r) {
    r += (3.1415-Conductivity(cell))*(3.1415-Conductivity(cell));
  }, diff);
  TEST_FLOATING_EQUALITY(0.,diff,tol);
}

}  // end anonymous namespace
