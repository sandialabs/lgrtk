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
#include "Omega_h_shape.hpp"
#include <Omega_h_array_ops.hpp>
#include <Omega_h_functors.hpp>
#include <Omega_h_metric.hpp>
#include <Omega_h_quality.hpp>
#include <Omega_h_teuchos.hpp>
#include <Omega_h_defines.hpp>
#include "LGRTestHelpers.hpp"

#include <FEMesh.hpp>
#include <Fields.hpp>
#include <FieldDB.hpp>

#include "CellTools.cpp"
#include "LowRmPotentialSolve.cpp"
#include "ExplicitFunctors.cpp"

#include "LGR_Types.hpp"
#include <limits>

namespace {

using std::endl;
using std::cout;

static constexpr int SpatialDim = 3;
typedef lgr::Fields<SpatialDim> Fields;

template<int SpatialDim>
inline int calcEdgesPerElement()
{
  int EdgesPerElement = 0;
  for (int i=1; i <= SpatialDim; i++) {
    EdgesPerElement += i;
  }
  return EdgesPerElement;
}

template<int SpatialDim>
void generate3DMesh(lgr::FEMesh<SpatialDim> &mesh, const size_t nx, const size_t ny, const size_t nz, 
                    Teuchos::FancyOStream &out, bool &success)
{
  const int meshDim = mesh.numDim;

  cout << "created FEMesh with dimension = " << meshDim << endl;
  TEST_EQUALITY_CONST(meshDim, 3);

  TEST_EQUALITY(meshDim, mesh.omega_h_mesh->dim() );

  mesh.machine = lgr::getCommMachine();

  mesh.resetSizes();

 // Element counts:
  const size_t elem_count = mesh.omega_h_mesh->nelems();
  cout <<"elem_count = " << elem_count << endl;
  TEST_EQUALITY(elem_count, 6*nx*ny*nz ); // each brick is six tets

  const size_t face_count = Omega_h::FACE <= mesh.omega_h_mesh->dim() ? mesh.omega_h_mesh->nfaces() : 0;

  // list of total nodes on the rank
  size_t node_count = mesh.omega_h_mesh->nverts();
  cout << "nverts = " << node_count << endl;
  TEST_EQUALITY(node_count, (nx+1)*(ny+1)*(nz+1) );

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
}

template<int SpatialDim>
void generate3DCellWorkset(lgr::FEMesh<SpatialDim> &mesh, lgr::CellTools::PhysPointsView &cellWorkset,
                           Teuchos::FancyOStream &, bool &)
{
  const auto elem_node_ids = mesh.elem_node_ids;
  const auto node_coords = mesh.node_coords;

  const int nodesPerCell = SpatialDim + 1;
  auto      numCells = cellWorkset.extent(0);

  Kokkos::parallel_for(
      Kokkos::RangePolicy<int>(0, numCells),
      LAMBDA_EXPRESSION(int cellOrdinal) {
        for (int nodeOrdinal = 0; nodeOrdinal < nodesPerCell; nodeOrdinal++) {
          const int n = elem_node_ids(cellOrdinal, nodeOrdinal);          
          for (int d = 0; d < SpatialDim; d++) {
            cellWorkset(cellOrdinal, nodeOrdinal, d) = node_coords(n, d);
          }
        }
      },
      "initialize workset");
}

TEUCHOS_UNIT_TEST(ElementHelperReplacement, Cell_Volume)
{
  constexpr double tol = 1e-12;
  lgr::FEMesh<SpatialDim> mesh;

  auto libOmegaH = lgr::getLibraryOmegaH();

  const size_t nx = 1;
  const size_t ny = 1;
  const size_t nz = 1; 

  auto omega_h_mesh = build_box(libOmegaH->world(),OMEGA_H_SIMPLEX,1,2,3,nx,ny,nz);
  mesh.omega_h_mesh = &omega_h_mesh;

  generate3DMesh(mesh,nx,ny,nz,out,success);

  const size_t elem_count = mesh.omega_h_mesh->nelems();
  const auto elem_node_ids = mesh.elem_node_ids;
  const auto node_coords = mesh.node_coords;
  const int ElemNodeCount = Fields::ElemNodeCount;

  lgr::CellTools::PhysPointsView cellWorkset("cell workset", elem_count, ElemNodeCount, SpatialDim);
  generate3DCellWorkset(mesh,cellWorkset,out,success);
  const auto numCells = cellWorkset.extent(0);

  lgr::Scalar totalVolumeElementHelper = 0;
  
  lgr::CellTools::FusedJacobianDetView elementHelperVolume("jacobian det.", elem_count);

  // Calculate cell volumes using dot4 from ElementHelpers.hpp
  Kokkos::parallel_reduce( elem_count, LAMBDA_EXPRESSION(const int ielem, lgr::Scalar &localVolume)
      {
        lgr::Scalar x[ElemNodeCount], y[ElemNodeCount], z[ElemNodeCount];
        lgr::Scalar grad_x[ElemNodeCount], grad_y[ElemNodeCount], grad_z[ElemNodeCount];

        // Position:   
        for (int i = 0; i < ElemNodeCount; ++i) {
          const int n = elem_node_ids(ielem, i);
          //use coordinates at next state
          x[i] = node_coords(n, 0);
          y[i] = node_coords(n, 1);
          z[i] = node_coords(n, 2);
        }
       
        // Gradient:
        lgr::comp_grad(x, y, z, grad_x, grad_y, grad_z);
       
        // lagrangian conservation of mass
        const lgr::Scalar vol = lgr::dot4(x, grad_x);
        elementHelperVolume(ielem) = vol;
        localVolume = vol;
      }, totalVolumeElementHelper);

  cout << "Total Volume from Element Helpers = " << totalVolumeElementHelper << endl;

  // Calculate cell volumes using CellTools
  lgr::CellTools::FusedJacobianView<SpatialDim> jacobian("jacobian", numCells);
  lgr::CellTools::FusedJacobianDetView jacobianDet("jacobian det.", numCells);
  lgr::CellTools::FusedJacobianDetView cellMeasure("cell measure", numCells);

  lgr::CellTools::setFusedJacobian<SpatialDim>(jacobian,cellWorkset);
  lgr::CellTools::setFusedJacobianDet<SpatialDim>(jacobianDet,jacobian);
  lgr::CellTools::getCellMeasure<SpatialDim>(cellMeasure,jacobianDet);

  lgr::Scalar totalVolumeCellTools = 0;

  Kokkos::parallel_reduce( numCells, LAMBDA_EXPRESSION(const int cellOrdinal, lgr::Scalar &localVolume)
      {  
        const lgr::Scalar vol = cellMeasure(cellOrdinal);
        localVolume = vol;
      }, totalVolumeCellTools);
  
  cout << "Total Volume from Cell Tools = " << totalVolumeCellTools << endl;

  TEST_FLOATING_EQUALITY(totalVolumeElementHelper, totalVolumeCellTools, tol);

  // Copy to Host
  lgr::CellTools::FusedJacobianDetView::HostMirror elementHelperVolumeHost = 
                                                     Kokkos::create_mirror_view( elementHelperVolume );
  lgr::CellTools::FusedJacobianDetView::HostMirror cellMeasureHost =
                                                     Kokkos::create_mirror_view( cellMeasure );

  Kokkos::deep_copy( elementHelperVolumeHost, elementHelperVolume );
  Kokkos::deep_copy( cellMeasureHost, cellMeasure );

  // Perform Test
  for (size_t cellOrdinal=0; cellOrdinal<elem_count; cellOrdinal++)
  {
    cout << "cell = " << cellOrdinal 
        << "; Volume from Element Helpers = " << elementHelperVolumeHost(cellOrdinal)
        << "; Volume from Cell Tools= " << cellMeasureHost(cellOrdinal)
        << endl;
    TEST_FLOATING_EQUALITY(elementHelperVolumeHost(cellOrdinal), cellMeasureHost(cellOrdinal), tol);
  }

}

TEUCHOS_UNIT_TEST(ElementHelperReplacement, Fused_Cell_Gradients)
{
  constexpr double tol = 1e-12;
  lgr::FEMesh<SpatialDim> mesh;

  auto libOmegaH = lgr::getLibraryOmegaH();

  const size_t nx = 1;
  const size_t ny = 1;
  const size_t nz = 1; 

  auto omega_h_mesh = build_box(libOmegaH->world(),OMEGA_H_SIMPLEX,1,2,3,nx,ny,nz);
  mesh.omega_h_mesh = &omega_h_mesh;

  generate3DMesh(mesh,nx,ny,nz,out,success);

  const size_t elem_count = mesh.omega_h_mesh->nelems();
  const auto elem_node_ids = mesh.elem_node_ids;
  const auto node_coords = mesh.node_coords;
  const int ElemNodeCount = Fields::ElemNodeCount;

  lgr::CellTools::PhysPointsView cellWorkset("cell workset", elem_count, ElemNodeCount, SpatialDim);
  generate3DCellWorkset(mesh,cellWorkset,out,success);
  const auto numCells = cellWorkset.extent(0);

  const int numFields = ElemNodeCount;
  lgr::CellTools::PhysCellGradientView elementHelperGradients("gradients from ElementHelpers.cpp", 
                                                               elem_count, numFields, SpatialDim);

  // Calculate Gradients using functions in ElementHelpers.hpp
  Kokkos::parallel_for( elem_count, LAMBDA_EXPRESSION(const int ielem)
      {
        lgr::Scalar x[ElemNodeCount], y[ElemNodeCount], z[ElemNodeCount];
        lgr::Scalar grad_x[ElemNodeCount], grad_y[ElemNodeCount], grad_z[ElemNodeCount];

        // Position:   
        for (int i = 0; i < ElemNodeCount; ++i) {
          const int n = elem_node_ids(ielem, i);
          //use coordinates at next state
          x[i] = node_coords(n, 0);
          y[i] = node_coords(n, 1);
          z[i] = node_coords(n, 2);
        }
       
        // Gradient:
        lgr::comp_grad(x, y, z, grad_x, grad_y, grad_z);
        
        for (int i = 0; i < ElemNodeCount; ++i) {
          elementHelperGradients(ielem, i, 0) = grad_x[i];
          elementHelperGradients(ielem, i, 1) = grad_y[i];
          elementHelperGradients(ielem, i, 2) = grad_z[i];
        }
      
      });

  // Calculate gradients using CellTools
  lgr::CellTools::FusedJacobianView<SpatialDim> jacobian("jacobian", numCells);
  lgr::CellTools::FusedJacobianView<SpatialDim> jacobianInv("jacobian inverse", numCells);
  lgr::CellTools::FusedJacobianDetView jacobianDet("jacobian det.", numCells);
  lgr::CellTools::FusedJacobianDetView cellMeasure("cell measure", numCells);

  lgr::CellTools::setFusedJacobian<SpatialDim>(jacobian,cellWorkset);
  lgr::CellTools::setFusedJacobianDet<SpatialDim>(jacobianDet,jacobian);
  lgr::CellTools::setFusedJacobianInv<SpatialDim>(jacobianInv,jacobian);
  lgr::CellTools::getCellMeasure<SpatialDim>(cellMeasure,jacobianDet);

  lgr::CellTools::PhysCellGradientView cellGradients("cell gradients", numCells, numFields, SpatialDim);
  lgr::CellTools::getPhysicalGradients(cellGradients, jacobianInv);

  lgr::CellTools::PhysCellGradientView 
    integratedCellGradients("integrated cell gradients", numCells, numFields, SpatialDim);

  Kokkos::parallel_for( numCells, LAMBDA_EXPRESSION(const int cellOrdinal)
      {  
        for (int nodeOrdinal=0; nodeOrdinal<numFields; nodeOrdinal++) {
          for (int dimOrdinal=0; dimOrdinal<SpatialDim; dimOrdinal++) {
            integratedCellGradients(cellOrdinal,nodeOrdinal,dimOrdinal) = 
                cellMeasure(cellOrdinal)*cellGradients(cellOrdinal,nodeOrdinal,dimOrdinal);
          }
        }
      });
 
  // Copy to Host
  lgr::CellTools::PhysCellGradientView::HostMirror elementHelperGradientsHost = 
                                                     Kokkos::create_mirror_view( elementHelperGradients );
  lgr::CellTools::PhysCellGradientView::HostMirror integratedCellGradientsHost =
                                                     Kokkos::create_mirror_view( integratedCellGradients );

  Kokkos::deep_copy( elementHelperGradientsHost, elementHelperGradients );
  Kokkos::deep_copy( integratedCellGradientsHost, integratedCellGradients );

  // Perform Test
  for (size_t cellOrdinal=0; cellOrdinal<elem_count; cellOrdinal++) {
    cout << "Testing gradients for cell = " << cellOrdinal << endl;
    for (int nodeOrdinal=0; nodeOrdinal<numFields; nodeOrdinal++) {
      for (int dimOrdinal=0; dimOrdinal<SpatialDim; dimOrdinal++) {
        TEST_FLOATING_EQUALITY(elementHelperGradientsHost(cellOrdinal,nodeOrdinal,dimOrdinal), 
                               integratedCellGradientsHost(cellOrdinal,nodeOrdinal,dimOrdinal), tol);
      }
    }
  }

}

TEUCHOS_UNIT_TEST(ElementHelperReplacement, Edge_Length)
{
  constexpr double tol = 1e-12;
  lgr::FEMesh<SpatialDim> mesh;

  auto libOmegaH = lgr::getLibraryOmegaH();

  const size_t nx = 1;
  const size_t ny = 1;
  const size_t nz = 1; 

  auto omega_h_mesh = build_box(libOmegaH->world(),OMEGA_H_SIMPLEX,1,2,3,nx,ny,nz);
  mesh.omega_h_mesh = &omega_h_mesh;

  generate3DMesh(mesh,nx,ny,nz,out,success);

  const size_t elem_count = mesh.omega_h_mesh->nelems();
  const auto elem_node_ids = mesh.elem_node_ids;
  const auto node_coords = mesh.node_coords;
  const int ElemNodeCount = Fields::ElemNodeCount;
  auto edgeLengths = Omega_h::measure_edges_real(mesh.omega_h_mesh);

  //Couldn't find this anywhere, perhaps it should be added to FEMesh
  const int ElemEdgeCount = calcEdgesPerElement<SpatialDim>();
  cout << "nedges = " << ElemEdgeCount << endl;

  auto elemToEdge = mesh.omega_h_mesh->ask_down(mesh.omega_h_mesh->dim(),Omega_h::EDGE).ab2b;

  lgr::CellTools::PhysPointsView cellWorkset("cell workset", elem_count, ElemNodeCount, SpatialDim);
  generate3DCellWorkset(mesh,cellWorkset,out,success);
  
  lgr::CellTools::FusedJacobianDetView maxEdgeElementHelpers("max edge length from element helpers", elem_count);
  lgr::CellTools::FusedJacobianDetView minEdgeElementHelpers("min edge length from element helpers", elem_count);

  // Calculate Max and Min edges using the function in ElementHelpers.hpp
  Kokkos::parallel_for( elem_count, LAMBDA_EXPRESSION(const int ielem)
      {
        lgr::Scalar x[ElemNodeCount], y[ElemNodeCount], z[ElemNodeCount];

        // Position:   
        for (int i = 0; i < ElemNodeCount; ++i) {
          const int n = elem_node_ids(ielem, i);
          //use coordinates at next state
          x[i] = node_coords(n, 0);
          y[i] = node_coords(n, 1);
          z[i] = node_coords(n, 2);
        }
       
        maxEdgeElementHelpers(ielem) = lgr::maxEdgeLength(x, y, z);
        minEdgeElementHelpers(ielem) = lgr::minEdgeLength(x, y, z);
        
      });

  lgr::CellTools::FusedJacobianDetView maxEdgeOmegaH("max edge length from omega_h", elem_count);
  lgr::CellTools::FusedJacobianDetView minEdgeOmegaH("min edge length from omega_h", elem_count);

  // Calculate Max and Min edges using Omega_h
  Kokkos::parallel_for( elem_count, LAMBDA_EXPRESSION(const int ielem)
      {
        Omega_h::LO edgeOrdinal = elemToEdge[ielem*ElemEdgeCount];
        maxEdgeOmegaH(ielem) = edgeLengths[edgeOrdinal];
        minEdgeOmegaH(ielem) = edgeLengths[edgeOrdinal];
        for (Omega_h::LO i = 0; i < ElemEdgeCount; i++) { 
          edgeOrdinal = elemToEdge[ielem*ElemEdgeCount + i];
          if (edgeLengths[edgeOrdinal] > maxEdgeOmegaH(ielem)) maxEdgeOmegaH(ielem) = edgeLengths[edgeOrdinal];
          if (edgeLengths[edgeOrdinal] < minEdgeOmegaH(ielem)) minEdgeOmegaH(ielem) = edgeLengths[edgeOrdinal];
        }
        
      });

  // Copy to Host
  lgr::CellTools::FusedJacobianDetView::HostMirror maxEdgeElementHelpersHost = 
                                                     Kokkos::create_mirror_view( maxEdgeElementHelpers );
  lgr::CellTools::FusedJacobianDetView::HostMirror minEdgeElementHelpersHost =
                                                     Kokkos::create_mirror_view( minEdgeElementHelpers );

  lgr::CellTools::FusedJacobianDetView::HostMirror maxEdgeOmegaHHost = 
                                                     Kokkos::create_mirror_view( maxEdgeOmegaH );
  lgr::CellTools::FusedJacobianDetView::HostMirror minEdgeOmegaHHost =
                                                     Kokkos::create_mirror_view( minEdgeOmegaH );

  Kokkos::deep_copy( maxEdgeElementHelpersHost, maxEdgeElementHelpers );
  Kokkos::deep_copy( minEdgeElementHelpersHost, minEdgeElementHelpers );
  Kokkos::deep_copy( maxEdgeOmegaHHost, maxEdgeOmegaH );
  Kokkos::deep_copy( minEdgeOmegaHHost, minEdgeOmegaH );

  // Perform Test
  for (size_t cellOrdinal=0; cellOrdinal<elem_count; cellOrdinal++)
  {
    cout << "cell = " << cellOrdinal 
        << "; Maximum Edge Length from Element Helpers = " << maxEdgeElementHelpersHost(cellOrdinal)
        << "; Maximum Edge Length from Omega_h = " << maxEdgeOmegaHHost(cellOrdinal)
        << endl;
    TEST_FLOATING_EQUALITY(maxEdgeElementHelpersHost(cellOrdinal), maxEdgeOmegaHHost(cellOrdinal), tol);
    cout << "cell = " << cellOrdinal 
        << "; Minimum Edge Length from Element Helpers = " << minEdgeElementHelpersHost(cellOrdinal)
        << "; Minimum Edge Length from Omega_h = " << minEdgeOmegaHHost(cellOrdinal)
        << endl;
    TEST_FLOATING_EQUALITY(minEdgeElementHelpersHost(cellOrdinal), minEdgeOmegaHHost(cellOrdinal), tol);
  }

}

}  // end anonymous namespace
