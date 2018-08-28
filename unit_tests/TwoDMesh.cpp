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

#include <Kokkos_Core.hpp>

#include <Kokkos_View.hpp>
#include <Kokkos_Parallel_Reduce.hpp>

#include <Omega_h_array.hpp>
#include <Omega_h_assoc.hpp>
#include <Omega_h_build.hpp>
#include <Omega_h_comm.hpp>
#include <Omega_h_defines.hpp>
#include <Omega_h_library.hpp>
#include <Omega_h_map.hpp>
#include <Omega_h_mark.hpp>
#include <Omega_h_mesh.hpp>
#include <Teuchos_LocalTestingHelpers.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCPDecl.hpp>
#include <Teuchos_UnitTestHelpers.hpp>
#include <cstdlib>

#include <cmath>
#include <complex>
#include <cstdlib>
#include <iostream>
#include <ostream>
#include <string>

#include "LGR_Types.hpp"
#include "LGRLambda.hpp"
#include "FEMesh.hpp"
#include "Fields.hpp"
#include "ParallelComm.hpp"
#include "LGRTestHelpers.hpp"

namespace {

TEUCHOS_UNIT_TEST(TwoDMesh, Create)
{

  using std::cout;
  using std::endl;

  static constexpr int SpatialDim = 2;
  typedef lgr::Fields<SpatialDim> Fields;


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

  const size_t nx = 4;
  const size_t ny = 4;

  auto omega_h_mesh = build_box(libOmegaH->world(),OMEGA_H_SIMPLEX,1,1,0,nx,ny,0);
  lgr::FEMesh<SpatialDim> mesh;

  const int meshDim = mesh.numDim;

  cout << "created FEMesh with dimension = " << meshDim << std::endl;
  TEST_EQUALITY_CONST(meshDim, 2);

  {
    constexpr double tol = 1e-12;
    mesh.omega_h_mesh = &omega_h_mesh;
    TEST_EQUALITY(meshDim, mesh.omega_h_mesh->dim() );

    mesh.machine = lgr::getCommMachine();

    mesh.resetSizes();

   // Element counts:
    const int elem_count = mesh.omega_h_mesh->nelems();
    cout <<"elem_count = " << elem_count << endl;
    TEST_EQUALITY(elem_count, 2*nx*ny ); // each quad is two tris

    const int face_count = Omega_h::FACE <= mesh.omega_h_mesh->dim() ? mesh.omega_h_mesh->nfaces() : 0;

    // list of total nodes on the rank
    const int node_count = mesh.omega_h_mesh->nverts();
    cout << "nverts = " << node_count << endl;

    TEST_EQUALITY(node_count, (nx+1)*(ny+1) );


   // Allocate the initial arrays. Note that the mesh.reAlloc() function does the resize
   if ( node_count ) {
      mesh.node_coords = typename Fields::node_coords_type( "node_coords", mesh.geom_layout );
    }
    if ( elem_count ) {
      mesh.elem_node_ids = typename Fields::elem_node_ids_type( "elem_node_ids", elem_count );
    }
    if ( face_count ) {
      mesh.face_node_ids = typename Fields::face_node_ids_type( "face_node_ids", face_count );
    }
    mesh.updateMesh();

    const auto elem_node_ids = mesh.elem_node_ids;
    const auto node_coords = mesh.node_coords;

    Kokkos::MinMaxScalar<lgr::Scalar> xminmax;
    Kokkos::MinMax<lgr::Scalar, Kokkos::DefaultExecutionSpace> xreducer(xminmax);

    Kokkos::parallel_reduce( node_count,
                     LAMBDA_EXPRESSION(const int n,Kokkos::MinMaxScalar<lgr::Scalar> & localValue ) {
                        if(node_coords(n, 0) < localValue.min_val) localValue.min_val = node_coords(n, 0) ;
                        if(node_coords(n, 0) > localValue.max_val) localValue.max_val = node_coords(n, 0) ;

                     },
                     xreducer );

    Kokkos::MinMaxScalar<lgr::Scalar> yminmax;
    Kokkos::MinMax<lgr::Scalar, Kokkos::DefaultExecutionSpace> yreducer(yminmax);

    Kokkos::parallel_reduce( node_count,
                     LAMBDA_EXPRESSION(const int n,Kokkos::MinMaxScalar<lgr::Scalar> & localValue ) {
                        if(node_coords(n, 1) < localValue.min_val) localValue.min_val = node_coords(n, 1) ;
                        if(node_coords(n, 1) > localValue.max_val) localValue.max_val = node_coords(n, 1) ;

                     },
                     yreducer );


    TEST_FLOATING_EQUALITY(xminmax.min_val, 0., tol);
    TEST_FLOATING_EQUALITY(yminmax.min_val, 0., tol);
    TEST_FLOATING_EQUALITY(xminmax.max_val, 1., tol);
    TEST_FLOATING_EQUALITY(yminmax.max_val, 1., tol);

    lgr::Scalar totalArea = 0;
    Kokkos::parallel_reduce(elem_count, LAMBDA_EXPRESSION(const int e, lgr::Scalar &localArea) {
        const auto i1 = elem_node_ids(e,0);
        const auto i2 = elem_node_ids(e,1);
        const auto i3 = elem_node_ids(e,2);
        localArea += 0.5*std::abs( node_coords(i1, 0)*(node_coords(i2, 1) - node_coords(i3, 1)) +
                                   node_coords(i2, 0)*(node_coords(i3, 1) - node_coords(i1, 1)) +
                                   node_coords(i3, 0)*(node_coords(i1, 1) - node_coords(i2, 1)  ));
    },
    totalArea);
    cout << "totalArea = " << totalArea << endl;

    TEST_FLOATING_EQUALITY(totalArea, 1., tol);

    Omega_h::Read<Omega_h::I8> interiorMarks = Omega_h::mark_by_class_dim(mesh.omega_h_mesh, SpatialDim, SpatialDim);
    Omega_h::LOs               localOrdinals = Omega_h::collect_marked(interiorMarks);
    elementSets["1"] = localOrdinals;
  }
}

}  // end anonymous namespace
