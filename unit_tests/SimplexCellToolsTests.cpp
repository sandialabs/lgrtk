#include "Teuchos_UnitTestHarness.hpp"

#include "LGRTestHelpers.hpp"
#include "Basis.hpp"
#include "CellTools.hpp"
#include "Cubature.hpp"

#include "Kokkos_Core.hpp"

using namespace lgr;

namespace {
  typedef Kokkos::View<Scalar***, Layout, MemSpace>     PhysicalPointsView;// (C,P,D)
  typedef Kokkos::View<Scalar**, Layout, MemSpace>      RefPointsView;     // (P,D)
  typedef Kokkos::View<Scalar*,  MemSpace>      WeightsView;       // (P)
  typedef Kokkos::View<Scalar*,  MemSpace>      VectorView;       // (C)
  
  template<int spaceDim>
  void testVolumeMatchesDeterminant(PhysicalPointsView &cellWorkset, VectorView &expectedVolumes, Teuchos::FancyOStream &out, bool &success)
  {
    /*
      For simplices, it should be the case that the determinant of the Jacobian is a constant,
      and is linearly related to the volume.  The multiplier is 1 / d!, where d is the spatial dimension.
    */
    
    int numCells = cellWorkset.extent(0);
    TEST_EQUALITY(cellWorkset.extent(2), spaceDim); // sanity check on our input

    out << "Allocating jacobianDet.\n";
    VectorView jacobianDet("Jacobian determinants", numCells);

    out << "Allocating fusedJacobian.\n";
    CellTools::FusedJacobianView<spaceDim> fusedJacobian("fused Jacobian",numCells);
    
    out << "Calling setFusedJacobian.\n";
    CellTools::setFusedJacobian(fusedJacobian, cellWorkset);
    out << "Calling setFusedJacobianDet.\n";
    CellTools::setFusedJacobianDet(jacobianDet, fusedJacobian);

    double multiplier = 1.0;
    for (int d=0; d<spaceDim; d++)
    {
      multiplier /= (d+1);
    }
    
    out << "Creating mirror views.\n";
    VectorView::HostMirror jacobianDetHost = Kokkos::create_mirror_view( jacobianDet );
    VectorView::HostMirror expectedVolumesHost = Kokkos::create_mirror_view( expectedVolumes );
    out << "Copying data to host.\n";
    Kokkos::deep_copy( jacobianDetHost, jacobianDet );
    Kokkos::deep_copy( expectedVolumesHost, expectedVolumes );

    out << "Entering comparison loop.\n";
    double tol = 1e-14;
    for (int cellOrdinal=0; cellOrdinal<numCells; cellOrdinal++)
    {
      double volume = fabs(jacobianDetHost(cellOrdinal) * multiplier);
      TEST_FLOATING_EQUALITY(volume, expectedVolumesHost(cellOrdinal), tol);
    }
    
    VectorView cellMeasures("cell measures", numCells);
    CellTools::getCellMeasure<spaceDim>(cellMeasures, jacobianDet);
    
    testFloatingEquality(expectedVolumes, cellMeasures, tol, out, success);
  }

  void testWorksetMapToPhysicalFrame(PhysicalPointsView &cellWorkset, Teuchos::FancyOStream &out, bool &success)
  {
    /*
     In this simple test, we confirm that mapping the reference nodes to physical coordinates
     recovers the cell workset.  Caller provides a cell workset; we do the rest.
     */
    int numCells = cellWorkset.extent(0);
    int numPoints = cellWorkset.extent(1);
    int spaceDim = cellWorkset.extent(2); // C,P,D
    int nodeCount = spaceDim + 1; // simplex node count

    out << "Allocating reference nodes.\n";
    RefPointsView refNodes("reference nodes", nodeCount, spaceDim);

    out << "Constructing basis.\n";
    Basis basis(spaceDim);
    out << "Getting ref coords.\n";
    basis.getRefCoords(refNodes);
    // sanity check: make sure that the "points" (P) dimension of cellWorkset matches the node count
    TEST_EQUALITY(nodeCount,numPoints);

    out << "Allocating physPoints.\n";
    PhysicalPointsView physPoints("physical points", numCells,numPoints,spaceDim);
    out << "Calling mapToPhysicalFrame.\n";
    CellTools::mapToPhysicalFrame(physPoints, refNodes, cellWorkset);
    
    out << "Calling testFloatingEquality<>.\n";
    double tol=1e-15;
    testFloatingEquality<Scalar,PhysicalPointsView>(physPoints,cellWorkset,tol,out,success);
    out << "Returning from testWorksetMapToPhysicalFrame.\n";
  }
  
  void getUnitCubeDiscretization(PhysicalPointsView::HostMirror cellWorksetHost, VectorView::HostMirror cellVolumesHost) // as simplices (tetrahedra)
  {
    int spaceDim = 3;
    int numCells = 5;
    // to discretize a cube into 5 tetrahedra, we can take the four vertices
    // (1,1,1)
    // (0,0,1)
    // (0,1,0)
    // (1,0,0)
    // as an interior tetrahedron.  This is cell 0 below.  The remaining 4 cells can be determined
    // by selecting three of the above points (there are exactly 4 such combinations) and then selecting
    // from the remaining four vertices of the cube the one nearest the plane defined by those three points.
    // The remaining four vertices are:
    // (0,0,0)
    // (1,1,0)
    // (1,0,1)
    // (0,1,1)
    // These have coordinates equal to 1 minus the coordinates of the interior cell nodes.

    cellWorksetHost(0,0,0) = 1.0;
    cellWorksetHost(0,0,1) = 1.0;
    cellWorksetHost(0,0,2) = 1.0;
    cellWorksetHost(0,1,0) = 0.0;
    cellWorksetHost(0,1,1) = 0.0;
    cellWorksetHost(0,1,2) = 1.0;
    cellWorksetHost(0,2,0) = 0.0;
    cellWorksetHost(0,2,1) = 1.0;
    cellWorksetHost(0,2,2) = 0.0;
    cellWorksetHost(0,3,0) = 1.0;
    cellWorksetHost(0,3,1) = 0.0;
    cellWorksetHost(0,3,2) = 0.0;

    Scalar centroid[spaceDim];
    for (int cellOrdinal=1; cellOrdinal<numCells; cellOrdinal++) 
    {
      // let (cellOrdinal-1) be the point from cell 0 that we here exclude
      // initialize centroid to 0
      for (int d=0; d<spaceDim; d++)
      {
        centroid[d] = 0;
      }
      for (int nodeOrdinal=0; nodeOrdinal<3; nodeOrdinal++)
      {
        int offset = (nodeOrdinal >= cellOrdinal-1) ? 1 : 0;
        for (int d=0; d<spaceDim; d++)
        {
          cellWorksetHost(cellOrdinal,nodeOrdinal,d) = cellWorksetHost(0,nodeOrdinal+offset,d);
          centroid[d] += cellWorksetHost(cellOrdinal,nodeOrdinal,d) / 3.0;
        }
      }
      // find the non-interior-cell node nearest the centroid
      double leastDistance = 2.0;
      int closestNodeOrdinal = -1;
      for (int nodeOrdinal=0; nodeOrdinal<4; nodeOrdinal++)
      {
        double dist = 0.0;
        for (int d=0; d<spaceDim; d++)
        {
          dist += (centroid[d] - (1.0 - cellWorksetHost(0,nodeOrdinal,d))) * (centroid[d] - (1.0 - cellWorksetHost(0,nodeOrdinal,d)));
        }
        if (dist < leastDistance)
        {
          closestNodeOrdinal = nodeOrdinal;
          leastDistance = dist;
        }
      }
      for (int d=0; d<spaceDim; d++)
      {
        cellWorksetHost(cellOrdinal,3,d) = 1.0 - cellWorksetHost(0,closestNodeOrdinal,d);
      }
    }
    /*
    for (int cellOrdinal=0; cellOrdinal<numCells; cellOrdinal++)
    {
      using namespace std;
      cout << "cell " << cellOrdinal << " has vertices:\n";
      for (int nodeOrdinal=0; nodeOrdinal<numNodes; nodeOrdinal++)
      {
        cout << "(" << cellWorksetHost(cellOrdinal, nodeOrdinal, 0);
        cout << ", " << cellWorksetHost(cellOrdinal, nodeOrdinal, 1);
        cout << ", " << cellWorksetHost(cellOrdinal, nodeOrdinal, 2) << ")\n";
      }
    }*/
    //return cellWorkset;

    // set volumes

    // the first, interior tetrahedron has volume 1/3
    // the other four each have volume (2/3)/4 = 1/6
    cellVolumesHost(0) = 1.0 / 3.0;
    for (int cellOrdinal=1; cellOrdinal<numCells; cellOrdinal++)
    {
      cellVolumesHost(cellOrdinal) = 1.0 / 6.0;
    }
  }

  struct CellWorksetExample
  {
    PhysicalPointsView cellWorkset;
    VectorView         cellVolumes;
  };

  CellWorksetExample getWorksetExample(int spaceDim, Teuchos::FancyOStream &out)
  {
    int numNodes = spaceDim + 1;
    CellWorksetExample worksetExample;
    int numCells = 0;
    if (spaceDim == 1) numCells = 3;
    if (spaceDim == 2) numCells = 4;
    if (spaceDim == 3) numCells = 5;
    out << "Allocating host and device views for workset and expectedVolumes.\n";
    PhysicalPointsView cellWorkset("cell workset",numCells,numNodes,spaceDim);
    PhysicalPointsView::HostMirror cellWorksetHost = Kokkos::create_mirror_view( cellWorkset );
    VectorView expectedVolumes("expected volumes", numCells);
    VectorView::HostMirror expectedVolumesHost = Kokkos::create_mirror_view( expectedVolumes );
    if (spaceDim == 1)
    {
      // initialize cellWorkset: just an arbitrary set of physical nodes, typical of a FEM mesh
      cellWorksetHost(0,0,0) = 0.0;
      cellWorksetHost(0,1,0) = 3.0;
      cellWorksetHost(1,0,0) = 3.0;
      cellWorksetHost(1,1,0) = 4.5;
      cellWorksetHost(2,0,0) = 4.5;
      cellWorksetHost(2,1,0) = 5.0;

      for (int cellOrdinal=0; cellOrdinal<numCells; cellOrdinal++)
      {
        expectedVolumesHost(cellOrdinal) = fabs(cellWorksetHost(cellOrdinal,1,0) - cellWorksetHost(cellOrdinal,0,0));
      }
    }
    else if (spaceDim == 2)
    {
      out << "Initializing cellWorksetHost.\n";
      cellWorksetHost(0,0,0) = 0.0;
      cellWorksetHost(0,0,1) = 0.0;
      cellWorksetHost(0,1,0) = 0.0;
      cellWorksetHost(0,1,1) = 1.0;
      cellWorksetHost(0,2,0) = 1.0;
      cellWorksetHost(0,2,1) = 1.0;
      cellWorksetHost(1,0,0) = 0.0;
      cellWorksetHost(1,0,1) = 0.0;
      cellWorksetHost(1,1,0) = 1.0;
      cellWorksetHost(1,1,1) = 1.0;
      cellWorksetHost(1,2,0) = 1.0;
      cellWorksetHost(1,2,1) = 0.0;
      cellWorksetHost(2,0,0) = 1.0;
      cellWorksetHost(2,0,1) = 0.0;
      cellWorksetHost(2,1,0) = 1.0;
      cellWorksetHost(2,1,1) = 1.0;
      cellWorksetHost(2,2,0) = 2.0;
      cellWorksetHost(2,2,1) = 1.0;
      cellWorksetHost(3,0,0) = 1.0;
      cellWorksetHost(3,0,1) = 0.0;
      cellWorksetHost(3,1,0) = 2.0;
      cellWorksetHost(3,1,1) = 1.0;
      cellWorksetHost(3,2,0) = 2.0;
      cellWorksetHost(3,2,1) = 0.0;

      out << "Initializing expectedVolumesHost.\n";
      for (int cellOrdinal=0; cellOrdinal<numCells; cellOrdinal++)
      {
        expectedVolumesHost(cellOrdinal) = 0.5;
      }
    }
    else if (spaceDim == 3)
    {
      getUnitCubeDiscretization(cellWorksetHost,expectedVolumesHost);
    }
    // copy from host to device
    Kokkos::deep_copy( cellWorkset, cellWorksetHost );
    Kokkos::deep_copy( expectedVolumes, expectedVolumesHost );  // from host to device
    worksetExample.cellWorkset = cellWorkset;
    worksetExample.cellVolumes = expectedVolumes;
    out << "returning worksetExample.\n";
    return worksetExample;
  }
  
  TEUCHOS_UNIT_TEST( CellTools, MapToPhysicalFrameWorkset_1D )
  {
    out << "Entered MapToPhysicalFrameWorkset_1D\n";
    int spaceDim = 1;
    CellWorksetExample worksetExample = getWorksetExample(spaceDim,out);
    testWorksetMapToPhysicalFrame(worksetExample.cellWorkset, out, success);
    /*
    int numNodes = spaceDim + 1;
    int numCells = 3;
    
    out << "Allocating views on device and host.\n";
    PhysicalPointsView cellWorkset("cell workset",numCells,numNodes,spaceDim);
    PhysicalPointsView::HostMirror cellWorksetHost = Kokkos::create_mirror_view( cellWorkset );

    out << "Initializing view on host.\n";
    // initialize cellWorkset: just an arbitrary set of physical nodes, typical of a FEM mesh
    cellWorksetHost(0,0,0) = 0.0;
    cellWorksetHost(0,1,0) = 3.0;
    cellWorksetHost(1,0,0) = 3.0;
    cellWorksetHost(1,1,0) = 4.5;
    cellWorksetHost(2,0,0) = 4.5;
    cellWorksetHost(2,1,0) = 5.0;

    out << "Copying to device.\n";
    // copy to device
    Kokkos::deep_copy( cellWorkset, cellWorksetHost );

    out << "Calling test method.\n";
    testWorksetMapToPhysicalFrame(cellWorkset,out,success);*/
  }

  TEUCHOS_UNIT_TEST( CellTools, MapToPhysicalFrameWorkset_2D )
  {
    int spaceDim = 2;
    CellWorksetExample worksetExample = getWorksetExample(spaceDim,out);
    testWorksetMapToPhysicalFrame(worksetExample.cellWorkset, out, success);
/*    int numNodes = spaceDim + 1;
    int numCells = 4;
    PhysicalPointsView cellWorkset("cell workset",numCells,numNodes,spaceDim);
    // initialize cellWorkset: just an arbitrary set of physical nodes, typical of a FEM mesh
    cellWorkset(0,0,0) = 0.0;
    cellWorkset(0,0,1) = 0.0;
    cellWorkset(0,1,0) = 0.0;
    cellWorkset(0,1,1) = 1.0;
    cellWorkset(0,2,0) = 1.0;
    cellWorkset(0,2,1) = 1.0;
    cellWorkset(1,0,0) = 0.0;
    cellWorkset(1,0,1) = 0.0;
    cellWorkset(1,1,0) = 1.0;
    cellWorkset(1,1,1) = 1.0;
    cellWorkset(1,2,0) = 1.0;
    cellWorkset(1,2,1) = 0.0;
    cellWorkset(2,0,0) = 1.0;
    cellWorkset(2,0,1) = 0.0;
    cellWorkset(2,1,0) = 1.0;
    cellWorkset(2,1,1) = 1.0;
    cellWorkset(2,2,0) = 2.0;
    cellWorkset(2,2,1) = 1.0;
    cellWorkset(3,0,0) = 1.0;
    cellWorkset(3,0,1) = 0.0;
    cellWorkset(3,1,0) = 2.0;
    cellWorkset(3,1,1) = 1.0;
    cellWorkset(3,2,0) = 2.0;
    cellWorkset(3,2,1) = 0.0;
    testWorksetMapToPhysicalFrame(cellWorkset,out,success);*/
  }

  TEUCHOS_UNIT_TEST( CellTools, MapToPhysicalFrameWorkset_3D )
  {
    // let's discretize a cube
    const int spaceDim = 3;
    CellWorksetExample worksetExample = getWorksetExample(spaceDim,out);
    testWorksetMapToPhysicalFrame(worksetExample.cellWorkset, out, success);
  }

  TEUCHOS_UNIT_TEST( CellTools, DeterminantMatchesVolume_1D )
  {
    /*
      For simplices, it should be the case that the determinant of the Jacobian is a constant,
      and is linearly related to the volume.  The multiplier is 1 / d!, where d is the spatial dimension.
    */
    const int spaceDim = 1;
    CellWorksetExample worksetExample = getWorksetExample(spaceDim,out);
    testVolumeMatchesDeterminant<spaceDim>(worksetExample.cellWorkset, worksetExample.cellVolumes, out, success);

    /*int numNodes = spaceDim + 1;
    int numCells = 3;
    PhysicalPointsView cellWorkset("cell workset",numCells,numNodes,spaceDim);
    // initialize cellWorkset: just an arbitrary set of physical nodes, typical of a FEM mesh
    cellWorkset(0,0,0) = 0.0;
    cellWorkset(0,1,0) = 3.0;
    cellWorkset(1,0,0) = 3.0;
    cellWorkset(1,1,0) = 4.5;
    cellWorkset(2,0,0) = 4.5;
    cellWorkset(2,1,0) = 5.0;

    VectorView expectedVolumes("expected volumes", numCells);
    for (int cellOrdinal=0; cellOrdinal<numCells; cellOrdinal++)
    {
      expectedVolumes(cellOrdinal) = fabs(cellWorkset(cellOrdinal,1,0) - cellWorkset(cellOrdinal,0,0));
    }

    testVolumeMatchesDeterminant(cellWorkset, expectedVolumes, out, success);*/
  }

  TEUCHOS_UNIT_TEST( CellTools, DeterminantMatchesVolume_2D )
  {
    /*
      For simplices, it should be the case that the determinant of the Jacobian is a constant,
      and is linearly related to the volume.  The multiplier is 1 / d!, where d is the spatial dimension.
    */
    const int spaceDim = 2;
    CellWorksetExample worksetExample = getWorksetExample(spaceDim,out);
    testVolumeMatchesDeterminant<spaceDim>(worksetExample.cellWorkset, worksetExample.cellVolumes, out, success);
    /*
    int numNodes = spaceDim + 1;
    int numCells = 4;
    PhysicalPointsView cellWorkset("cell workset",numCells,numNodes,spaceDim);
    // initialize cellWorkset: just an arbitrary set of physical nodes, typical of a FEM mesh
    cellWorkset(0,0,0) = 0.0;
    cellWorkset(0,0,1) = 0.0;
    cellWorkset(0,1,0) = 0.0;
    cellWorkset(0,1,1) = 1.0;
    cellWorkset(0,2,0) = 1.0;
    cellWorkset(0,2,1) = 1.0;
    cellWorkset(1,0,0) = 0.0;
    cellWorkset(1,0,1) = 0.0;
    cellWorkset(1,1,0) = 1.0;
    cellWorkset(1,1,1) = 1.0;
    cellWorkset(1,2,0) = 1.0;
    cellWorkset(1,2,1) = 0.0;
    cellWorkset(2,0,0) = 1.0;
    cellWorkset(2,0,1) = 0.0;
    cellWorkset(2,1,0) = 1.0;
    cellWorkset(2,1,1) = 1.0;
    cellWorkset(2,2,0) = 2.0;
    cellWorkset(2,2,1) = 1.0;
    cellWorkset(3,0,0) = 1.0;
    cellWorkset(3,0,1) = 0.0;
    cellWorkset(3,1,0) = 2.0;
    cellWorkset(3,1,1) = 1.0;
    cellWorkset(3,2,0) = 2.0;
    cellWorkset(3,2,1) = 0.0;

    VectorView expectedVolumes("expected volumes", numCells);
    for (int cellOrdinal=0; cellOrdinal<numCells; cellOrdinal++)
    {
      expectedVolumes(cellOrdinal) = 0.5;
    }

    testVolumeMatchesDeterminant(cellWorkset, expectedVolumes, out, success);*/
  }

  TEUCHOS_UNIT_TEST( CellTools, DeterminantMatchesVolume_3D )
  {
    const int spaceDim = 3;
    CellWorksetExample worksetExample = getWorksetExample(spaceDim,out);
    testVolumeMatchesDeterminant<spaceDim>(worksetExample.cellWorkset, worksetExample.cellVolumes, out, success);
  }

  TEUCHOS_UNIT_TEST( CellTools, IntegrateCubic_1D)
  {
    // tests both the cubature rules and CellTools
    // define a cubic function
    auto f = LAMBDA_EXPRESSION (Scalar x) -> Scalar
    {
      return x * x * x;
    };
    const int spaceDim = 1;
    int degree = 3;
    int numCells = 3;
    CellWorksetExample worksetExample = getWorksetExample(spaceDim, out);
    PhysicalPointsView cellWorkset = worksetExample.cellWorkset;

    // integral of x^3 is x^4 / 4; integrated from 0 to 5 this is 5^4 / 4 = 156.25
    Scalar expectedIntegral = 156.25;
    
    typedef Kokkos::View<Scalar***, Layout, MemSpace>     View_3D;     // (C,P,D)
    typedef Kokkos::View<Scalar**, Layout, MemSpace>      View_2D;     // (P,D)
    typedef Kokkos::View<Scalar*,  MemSpace>      View_1D;     // (P)

    int numPoints = Cubature::getNumCubaturePoints(spaceDim, degree);

    out << "Allocating cubPoints, cubWeights containers on device.\n";
    View_2D cubPoints("ref. space cub. points", numPoints,spaceDim);
    View_1D cubWeights("cub. weights", numPoints);

    out << "Calling getCubature.\n";
    Cubature::getCubature(spaceDim, degree, cubPoints, cubWeights);
    
    out << "Allocating physCubPoints.\n";
    View_3D physCubPoints("physical cubature points", numCells,numPoints,spaceDim);
    out << "Calling mapToPhysicalFrame.\n";
    CellTools::mapToPhysicalFrame(physCubPoints, cubPoints, cellWorkset);

    View_2D physValues("function values at physical points", numCells, numPoints);
    for (int cellOrdinal=0; cellOrdinal<numCells; cellOrdinal++)
    Kokkos::parallel_for(numCells, LAMBDA_EXPRESSION(int cellOrd)  
    {
      for (int ptOrdinal=0; ptOrdinal<numPoints; ptOrdinal++)
      {
        Scalar x = physCubPoints(cellOrd,ptOrdinal,0);
        Scalar value = f(x);
        physValues(cellOrd,ptOrdinal) = value;
      }
    });

    Scalar integral = 0.0;
    CellTools::FusedJacobianView<spaceDim> fusedJacobian   ("fused Jacobian",numCells);
    CellTools::FusedJacobianDetView        fusedJacobianDet("fused Jacobian determinant",numCells);

    CellTools::setFusedJacobian(fusedJacobian,cellWorkset);
    CellTools::setFusedJacobianDet(fusedJacobianDet,fusedJacobian);

    Kokkos::parallel_reduce(numCells, LAMBDA_EXPRESSION(const int cellOrdinal, Scalar &localIntegral) {
     Scalar mySum = 0.0;
     for (int ptOrdinal=0; ptOrdinal<numPoints; ptOrdinal++)
     {
       mySum += physValues(cellOrdinal,ptOrdinal) * cubWeights(ptOrdinal) * fabs(fusedJacobianDet(cellOrdinal));
     }
     localIntegral += mySum;
    }, integral);
    /*
    for (int cellOrdinal=0; cellOrdinal<numCells; cellOrdinal++)
    {
      for (int ptOrdinal=0; ptOrdinal<numPoints; ptOrdinal++)
      {
        integral += physValues(cellOrdinal,ptOrdinal) * cubWeights(ptOrdinal) * fabs(fusedJacobianDet(cellOrdinal));
      }
    }*/

    Scalar tol = 1e-14;
    TEST_FLOATING_EQUALITY(integral, expectedIntegral, tol);
    out << "Integral is " << integral << "\n";
    TEST_FLOATING_EQUALITY(156.25,156.25,tol);
    TEST_FLOATING_EQUALITY(.15625,.15625,tol);
  }

  TEUCHOS_UNIT_TEST( CellTools, IntegrateCubic_2D)
  {
    // tests both the cubature rules and CellTools
    // define a cubic function
    auto f = LAMBDA_EXPRESSION (Scalar x, Scalar y) -> Scalar
    {
      return x * x * x + x * y + 3 * y;
    };
    int degree = 3;
    const int spaceDim = 2;
    int numCells = 4;
    CellWorksetExample worksetExample = getWorksetExample(spaceDim,out);
    PhysicalPointsView cellWorkset = worksetExample.cellWorkset;

    // integral in x is x^4 / 4 + x^2 y / 2 + 3 xy; integrated from 0 to 2 in x this is 16 / 4 + 2 y + 6 y = 4 + 8y
    // integral of this in y is 4y + 4 y^2; from 0 to 1 this is 8
    Scalar expectedIntegral = 8.0;
    
    typedef Kokkos::View<Scalar***, Layout, MemSpace>     View_3D;     // (C,P,D)
    typedef Kokkos::View<Scalar**, Layout, MemSpace>      View_2D;     // (P,D)
    typedef Kokkos::View<Scalar*,  MemSpace>      View_1D;     // (P)

    int numPoints = Cubature::getNumCubaturePoints(spaceDim, degree);

    View_2D cubPoints("ref. space cub. points", numPoints,spaceDim);
    View_1D cubWeights("cub. weights", numPoints);

    Cubature::getCubature(spaceDim, degree, cubPoints, cubWeights);

    View_3D physCubPoints("physical cubature points", numCells,numPoints,spaceDim);
    CellTools::mapToPhysicalFrame(physCubPoints, cubPoints, cellWorkset);

    View_2D physValues("function values at physical points", numCells, numPoints);
    Kokkos::parallel_for(numCells, LAMBDA_EXPRESSION(int cellOrdinal)
    {
      for (int ptOrdinal=0; ptOrdinal<numPoints; ptOrdinal++)
      {
        Scalar x = physCubPoints(cellOrdinal,ptOrdinal,0);
        Scalar y = physCubPoints(cellOrdinal,ptOrdinal,1);
        Scalar value = f(x,y);
        physValues(cellOrdinal,ptOrdinal) = value;
      }
    });

    Scalar integral = 0.0;
    CellTools::FusedJacobianView<spaceDim> fusedJacobian   ("fused Jacobian",numCells);
    CellTools::FusedJacobianDetView        fusedJacobianDet("fused Jacobian determinant",numCells);

    CellTools::setFusedJacobian(fusedJacobian,cellWorkset);
    CellTools::setFusedJacobianDet(fusedJacobianDet,fusedJacobian);

    Kokkos::parallel_reduce(numCells, LAMBDA_EXPRESSION(const int cellOrdinal, double &localIntegral) {
     Scalar mySum = 0.0;
     for (int ptOrdinal=0; ptOrdinal<numPoints; ptOrdinal++)
     {
       mySum += physValues(cellOrdinal,ptOrdinal) * cubWeights(ptOrdinal) * fabs(fusedJacobianDet(cellOrdinal));
     }
     localIntegral += mySum;
    }, integral);
    /*for (int cellOrdinal=0; cellOrdinal<numCells; cellOrdinal++)
    {
      for (int ptOrdinal=0; ptOrdinal<numPoints; ptOrdinal++)
      {
        integral += physValues(cellOrdinal,ptOrdinal) * cubWeights(ptOrdinal) * fabs(fusedJacobianDet(cellOrdinal));
      }
    }*/

    Scalar tol = 1e-14;
    TEST_FLOATING_EQUALITY(integral, expectedIntegral, tol);
  }

  TEUCHOS_UNIT_TEST( CellTools, IntegrateCubic_3D )
  {
    // tests both the cubature rules and CellTools
    // define a cubic function
    auto f = LAMBDA_EXPRESSION (Scalar x, Scalar y, Scalar z) -> Scalar
    {
      return x * x * x + x * y + 3 * y + z * z;
    };
    int degree = 3;
    int numCells = 5;
    const int spaceDim = 3;
    CellWorksetExample worksetExample = getWorksetExample(spaceDim,out);
    PhysicalPointsView cellWorkset = worksetExample.cellWorkset;

    // integral in x is x^4 / 4 + x^2 y / 2 + 3 xy + xz^2; integrated from 0 to 1 in x this is 1 / 4 + y / 2 + 3y + z^2 = .25 + 3.5y + z^2
    // integral of this in y is .25 y + 1.75 y^2 + y z^2; from 0 to 1 this is 2.0 + z^2
    // integral of this in z is 2 z + z^3 / 3; from 0 to 1 this is 2 + 1/3 = 7/3
    Scalar expectedIntegral = 7.0 / 3.0;
    
    typedef Kokkos::View<Scalar***, Layout, MemSpace>     View_3D;     // (C,P,D)
    typedef Kokkos::View<Scalar**, Layout, MemSpace>      View_2D;     // (P,D)
    typedef Kokkos::View<Scalar*,  MemSpace>      View_1D;     // (P)

    int numPoints = Cubature::getNumCubaturePoints(spaceDim, degree);

    View_2D cubPoints("ref. space cub. points", numPoints,spaceDim);
    View_1D cubWeights("cub. weights", numPoints);

    Cubature::getCubature(spaceDim, degree, cubPoints, cubWeights);

    View_3D physCubPoints("physical cubature points", numCells,numPoints,spaceDim);
    CellTools::mapToPhysicalFrame(physCubPoints, cubPoints, cellWorkset);

    View_2D physValues("function values at physical points", numCells, numPoints);
    Kokkos::parallel_for(numCells, LAMBDA_EXPRESSION(int cellOrdinal) { 
      for (int ptOrdinal=0; ptOrdinal<numPoints; ptOrdinal++)
      {
        Scalar x = physCubPoints(cellOrdinal,ptOrdinal,0);
        Scalar y = physCubPoints(cellOrdinal,ptOrdinal,1);
        Scalar z = physCubPoints(cellOrdinal,ptOrdinal,2);
        Scalar value = f(x,y,z);
        physValues(cellOrdinal,ptOrdinal) = value;
      }
    });

    Scalar integral = 0.0;
    
    CellTools::FusedJacobianView<spaceDim> fusedJacobian   ("fused Jacobian",numCells);
    CellTools::FusedJacobianDetView        fusedJacobianDet("fused Jacobian determinant",numCells);

    CellTools::setFusedJacobian(fusedJacobian,cellWorkset);
    CellTools::setFusedJacobianDet(fusedJacobianDet,fusedJacobian);
    
    Kokkos::parallel_reduce(numCells, LAMBDA_EXPRESSION(const int cellOrdinal, double &localIntegral) {
     Scalar mySum = 0.0;
     for (int ptOrdinal=0; ptOrdinal<numPoints; ptOrdinal++)
     {
       mySum += physValues(cellOrdinal,ptOrdinal) * cubWeights(ptOrdinal) * fabs(fusedJacobianDet(cellOrdinal));
     }
     localIntegral += mySum;
    }, integral);
    
    // DEBUGGING -- this won't run on GPUs
//    Scalar hostIntegral = 0.0;
//    for (int cellOrdinal=0; cellOrdinal<numCells; cellOrdinal++)
//    {
//      for (int ptOrdinal=0; ptOrdinal<numPoints; ptOrdinal++)
//      {
//        Scalar physValue = physValues(cellOrdinal,ptOrdinal);
//        Scalar weight    = cubWeights(ptOrdinal);
//        Scalar jacDet    = fusedJacobianDet(cellOrdinal);
//        
//        std::cout << physValue << " * " << weight << " * abs(" << jacDet << ")\n";
//        
//        hostIntegral += physValue * weight * fabs(jacDet);
//      }
//    }

    Scalar tol = 1e-14;
    TEST_FLOATING_EQUALITY(integral, expectedIntegral, tol);
  }
  
  TEUCHOS_UNIT_TEST( CellTools, Jacobian_1D )
  {
    const int spaceDim = 1;
    const int numFields = spaceDim + 1;
    const int numNodes  = numFields;
    int numCells = 2;
    PhysicalPointsView cellWorkset("cell workset", numCells, numNodes, spaceDim);
    Kokkos::parallel_for(numCells, LAMBDA_EXPRESSION(int cellOrdinal) {
      Scalar xLeft =  Scalar(cellOrdinal  ) / numCells;
      Scalar xRight = Scalar(cellOrdinal+1) / numCells;
      cellWorkset(cellOrdinal,0,0) = xLeft;
      cellWorkset(cellOrdinal,1,0) = xRight;
    });
    
    CellTools::FusedJacobianView<spaceDim> fusedJacobian         ("Jacobian",          numCells);
    CellTools::FusedJacobianView<spaceDim> expectedJacobian      ("expected Jacobian", numCells);
    
    // nodes are flipped (node 0 is the one on the right in ref. space)  --> negative Jacobian, with scaling h = 1 / numCells
    
    CellTools::setFusedJacobian(fusedJacobian, cellWorkset);
    Kokkos::parallel_for(numCells, LAMBDA_EXPRESSION(int cellOrdinal) {
      expectedJacobian(cellOrdinal)[0][0] = -1.0 / Scalar(numCells);
    });
    
    typedef Kokkos::View<Scalar***,  Layout, MemSpace>      View_3D;     // (C,D,D)
    // copy data to some scalar-valued views, so that we can use our testFloatingEquality()
    View_3D jacobianScalarView        ("Jacobian as Scalars",          numCells, spaceDim, spaceDim);
    View_3D expectedJacobianScalarView("expected Jacobian as Scalars", numCells, spaceDim, spaceDim);
    
    Kokkos::parallel_for(numCells, LAMBDA_EXPRESSION(int cellOrdinal) {
      for (int d1=0; d1<spaceDim; d1++)
        for (int d2=0; d2<spaceDim; d2++)
        {
          jacobianScalarView        (cellOrdinal,d1,d2) = fusedJacobian   (cellOrdinal)[d1][d2];
          expectedJacobianScalarView(cellOrdinal,d1,d2) = expectedJacobian(cellOrdinal)[d1][d2];
        }
    });
    
    double tol=1e-14;
    testFloatingEquality(expectedJacobianScalarView, jacobianScalarView, tol, out, success);
  }
  
  TEUCHOS_UNIT_TEST( CellTools, PhysicalGradients_1D )
  {
    const int spaceDim = 1;
    const int numFields = spaceDim + 1;
    const int numNodes  = numFields;
    int numCells = 2;
    PhysicalPointsView cellWorkset("cell workset", numCells, numNodes, spaceDim);
    Kokkos::parallel_for(numCells, LAMBDA_EXPRESSION(int cellOrdinal) {
      Scalar xLeft =  Scalar(cellOrdinal  ) / numCells;
      Scalar xRight = Scalar(cellOrdinal+1) / numCells;
      cellWorkset(cellOrdinal,0,0) = xLeft;
      cellWorkset(cellOrdinal,1,0) = xRight;
    });
    
    CellTools::FusedJacobianView<spaceDim> fusedJacobian         ("fused Jacobian",             numCells);
    CellTools::FusedJacobianView<spaceDim> fusedJacobianInverse  ("fused Jacobian inverse",     numCells);
    CellTools::FusedJacobianDetView        fusedJacobianDet      ("fused Jacobian determinant", numCells);
    
    CellTools::setFusedJacobian(fusedJacobian, cellWorkset);
    CellTools::setFusedJacobianDet(fusedJacobianDet, fusedJacobian);
    CellTools::setFusedJacobianInv(fusedJacobianInverse, fusedJacobian);
    
    CellTools::PhysCellGradientView cellGradients("Cell Gradients", numCells, numFields, spaceDim);
    
    CellTools::getPhysicalGradients(cellGradients, fusedJacobianInverse);
    
    CellTools::PhysCellGradientView expectedCellGradients("Expected Cell Gradients", numCells, numFields, spaceDim);

    Kokkos::parallel_for(numCells, LAMBDA_EXPRESSION(int cellOrdinal) {
      // fields are numbered same as the nodes, and at node 0 we expect gradient of -1/h; at node 1 we expect 1/h
      expectedCellGradients(cellOrdinal,0,0) = -Scalar(numCells);
      expectedCellGradients(cellOrdinal,1,0) =  Scalar(numCells); // 1/h = 1/(1/numCells) = numCells
    });
    
    double tol=1e-14;
    testFloatingEquality(expectedCellGradients, cellGradients, tol, out, success);
  }
} // namespace
