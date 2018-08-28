#include "LGRTestHelpers.hpp"
#include "Omega_h_build.hpp"
#include "Omega_h_map.hpp"

#include "Teuchos_UnitTestHarness.hpp"

#include "CrsMatrix.hpp"
#include "FEMesh.hpp"
#include "LowRmPotentialSolve.hpp"
#include "MatrixIO.hpp"
#include "MeshFixture.hpp"

#ifdef HAVE_VIENNA_CL
#include "ViennaSparseLinearProblem.hpp"
#endif

#ifdef HAVE_AMGX
#include "AmgXSparseLinearProblem.hpp"
#endif

#include <sstream>

using namespace lgr;
using namespace Omega_h;

namespace {
  enum MatrixSolver
  {
    NONE,
    AMGX,
    VIENNACL
  };
  
  struct ExactSolution
  {
    std::string exactSolution;
    std::string forcingFunction;
    int forcingQuadratureDegree;
    double lumpedCircuitValue; // assuming a unit box domain and unit diagonal conductivity, what is the value of K_{11}?
    // worth noting that this lumped circuit value is the exact solution, which is not exactly recoverable.  Nodal values are
    // exactly recovered for Poisson even on a lowest-order mesh, but in general integral values will have some error...
  };
  
  std::vector<ExactSolution> getExactPolynomialSolutions()
  {
    // try an exact solution corresponding to x^4 / 12 - x^2 / 2 --> forcing = x^2-1
    // OR, an exact solution corresponding to x^2 / 2            --> forcing = 1
    // OR, an exact soultion corresponding to x^3 / 6            --> forcing = x
    // OR, an exact soultion corresponding to x^4 / 12           --> forcing = x^2
    
    using namespace std;
    vector<ExactSolution> solutions;
    solutions.push_back({"x^4 / 12 - x^2 / 2", "x^2-1", 2, 37.0 / 144.0});
    solutions.push_back({"x^2 / 2",            "1",     0,  1.0 /   3.0});
    solutions.push_back({"x^3 / 6",            "x",     1,  1.0 /  20.0});
    solutions.push_back({"x^4 / 12",           "x^2",   2,  1.0 / 144.0});
    
    return solutions;
  }
  
  typedef Scalar                                                                       CoordinateScalar;
  typedef DefaultLocalOrdinal                                                          Ordinal;
  typedef int                                                                          RowMapEntryType;
  
#ifdef HAVE_VIENNA_CL
  typedef ViennaSparseLinearProblem<Ordinal>                    ViennaLinearProblem;
#endif

#ifdef HAVE_AMGX
  typedef AmgXSparseLinearProblem<Ordinal>                      AmgXLinearProblem;
#endif
  
  typedef CrsMatrix<Ordinal, RowMapEntryType>          CrsMatrix;
  typedef MatrixIO <Ordinal, RowMapEntryType>          MatrixIO;
  
  typedef Kokkos::View<Ordinal*,         MemSpace> OrdinalVector;
  typedef Kokkos::View<Scalar* ,         MemSpace> ScalarVector;
  typedef Kokkos::View<RowMapEntryType*, MemSpace> RowMapEntryTypeVector;

  Omega_h::LOs getBoundaryNodes(Teuchos::RCP<Omega_h::Mesh> mesh)
  {
    int spaceDim  = mesh->dim();
    int vertexDim = 0;
    Omega_h::Read<I8> interiorMarks = Omega_h::mark_by_class_dim(mesh.get(), vertexDim, spaceDim);
    Omega_h::Read<I8> boundaryMarks = Omega_h::invert_marks(interiorMarks);
    Omega_h::LOs localOrdinals = Omega_h::collect_marked(boundaryMarks);
    
    return localOrdinals;
  }
  
  //! returns all nodes matching x=0 on the boundary of the provided mesh
  Omega_h::LOs getBoundaryNodes_x0(Teuchos::RCP<Omega_h::Mesh> mesh)
  {
    int spaceDim  = mesh->dim();
    TEUCHOS_TEST_FOR_EXCEPTION(spaceDim != 3, std::invalid_argument, "This method only implemented for 3D right now");
    
    // because of the way that build_box does things, the x=0 nodes end up on a face which has label (2,12); the x=1 nodes end up with label (2,14)
    const int vertexDim = 0;
    const int faceDim = spaceDim-1;
    Omega_h::Read<I8> x0Marks = Omega_h::mark_class_closure(mesh.get(), vertexDim, faceDim, 12);
    Omega_h::LOs localOrdinals = Omega_h::collect_marked(x0Marks);
    
    return localOrdinals;
  }
  
  //! returns all nodes matching x=1 on the boundary of the provided mesh
  Omega_h::LOs getBoundaryNodes_x1(Teuchos::RCP<Omega_h::Mesh> mesh)
  {
    int spaceDim  = mesh->dim();
    TEUCHOS_TEST_FOR_EXCEPTION(spaceDim != 3, std::invalid_argument, "This method only implemented for 3D right now");
    
    // because of the way that build_box does things, the x=0 nodes end up on a face which has label (2,12); the x=1 nodes end up with label (2,14)
    const int vertexDim = 0;
    const int faceDim = spaceDim-1;
    Omega_h::Read<I8> x1Marks = Omega_h::mark_class_closure(mesh.get(), vertexDim, faceDim, 14);
    Omega_h::LOs localOrdinals = Omega_h::collect_marked(x1Marks);
    
    return localOrdinals;
  }
  
  Teuchos::RCP<Omega_h::Mesh> getBoxMesh(int spaceDim, int meshWidth, Scalar x_scaling = 1.0, Scalar y_scaling = -1.0, Scalar z_scaling = -1.0)
  {
    if (y_scaling == -1.0) y_scaling = x_scaling;
    if (z_scaling == -1.0) z_scaling = y_scaling;
    
    int nx = 0, ny = 0, nz = 0;
    if (spaceDim == 1)
    {
      nx = meshWidth;
    }
    else if (spaceDim == 2)
    {
      nx = meshWidth;
      ny = meshWidth;
    }
    else if (spaceDim == 3)
    {
      nx = meshWidth;
      ny = meshWidth;
      nz = meshWidth;
    }
    
    Teuchos::RCP<Omega_h::Library> libOmegaH = getLibraryOmegaH();
    
    auto omega_h_mesh = Teuchos::rcp( new Mesh(
          build_box(libOmegaH->world(), OMEGA_H_SIMPLEX, x_scaling,y_scaling,z_scaling,nx,ny,nz) ) );
    return omega_h_mesh;
  }
  
  double getExpectedK11(int meshWidth, std::string solnExpr)
  {
    /*
     Here, we compute the expected *discrete* K11 value for a 1D mesh with nodally exact solution.
     
     It turns out that this works in 2D and 3D, too, at least for the meshes returned by build_box,
     if the exact solution varies only in x.
     */
    double expected_K11 = 0.0;
    {
      Omega_h::Write<Scalar> x_coords(meshWidth+1, 0.0);
      ScalarVector x_values("x values",meshWidth+1);
      Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,meshWidth + 1), LAMBDA_EXPRESSION(int xNodeOrdinal)
      {
        x_coords[xNodeOrdinal] = (double)xNodeOrdinal / (double)meshWidth;
      }, "fill x coords");
      
      evaluateNodalExpression(solnExpr, 1, x_coords, x_values);
      
      Kokkos::parallel_reduce("K11: determine expected discrete value",
          Kokkos::RangePolicy<int>(0,meshWidth), LAMBDA_EXPRESSION(int cellOrdinal, Scalar &localIntegral)
      {
        Scalar h = x_coords[cellOrdinal+1] - x_coords[cellOrdinal];
        Scalar gradient = (x_values(cellOrdinal+1) - x_values(cellOrdinal))/h;
        localIntegral += gradient * gradient * h;
      }, expected_K11);
    }
    return expected_K11;
  }
  
  template<int spaceDim>
  LowRmPotentialSolve<spaceDim> getLowRmPotentialSolveExample(Teuchos::RCP<Omega_h::Mesh> meshOmegaH)
  {
    using DefaultFields = Fields<spaceDim>;
    
    Teuchos::RCP<Omega_h::Library> libOmegaH = getLibraryOmegaH();
    
    // it should be that the only parts of fields we care about are the Omega_h mesh and a couple
    // specific fields: conductivity and conductors.  We leave setting those latter to individual tests,
    // but we will create a simple hypercube mesh here with a handful of elements
    
    FEMesh<spaceDim> mesh;
    {
      // hopefully the following is enough to get things set up in the FEMesh
      
      mesh.omega_h_mesh = meshOmegaH.get();
      if(mesh.numDim != mesh.omega_h_mesh->dim() )
        throw std::runtime_error("FEMesh spatial dim template parameter must match Omega_h spatial dimension.");

      mesh.machine = getCommMachine();
      
      mesh.resetSizes();
      
      // Element counts:
      
      const size_t elem_count = mesh.omega_h_mesh->nelems();
      const size_t face_count = Omega_h::FACE <= mesh.omega_h_mesh->dim() ? mesh.omega_h_mesh->nfaces() : 0;
      
      //! list of total nodes on the rank
      size_t node_count = mesh.omega_h_mesh->nverts();
      
      // Allocate the initial arrays. Note that the mesh.reAlloc() function does the resize
      if ( node_count ) {
        mesh.node_coords = typename DefaultFields::node_coords_type( "node_coords", mesh.geom_layout );
      }
      
      if ( elem_count ) {
        mesh.elem_node_ids = typename DefaultFields::elem_node_ids_type( "elem_node_ids", elem_count );
	mesh.elem_face_ids = typename DefaultFields::elem_face_ids_type( "elem_face_ids", elem_count );
	mesh.elem_face_orientations = typename DefaultFields::elem_face_orient_type( "elem_face_orientations", elem_count );
      }
      if ( face_count ) {
        mesh.face_node_ids = typename DefaultFields::face_node_ids_type( "face_node_ids", face_count );
      }
      
      mesh.updateMesh();
    }
    
    Teuchos::ParameterList emptyParamList;
    auto fields = Teuchos::rcp( new DefaultFields(mesh, emptyParamList) );
    
    LowRmPotentialSolve<spaceDim> solver(emptyParamList, fields, getCommMachine());
    solver.setConductivity(solver.getConstantConductivity(1.0));
    
    return solver;
  }
  
  template<int spaceDim>
  void testAssembledSystemMatchesSolution(LowRmPotentialSolve<spaceDim> &solve,
                                          typename LowRmPotentialSolve<spaceDim>::ScalarMultiVector &expectedSolution,
                                          Teuchos::FancyOStream &out, bool &success, double tol=1e-12)
  {
    auto matrix = solve.getMatrix();
    auto rhs    = solve.getRHS();
    
    // deal with the fact that we have 2D views when so far we only support 1D in CrsMatrix::Apply
    int numNodes = rhs.extent(1);
    
    typename LowRmPotentialSolve<spaceDim>::ScalarVector soln1D = Kokkos::subview (expectedSolution, 0, Kokkos::ALL());
    typename LowRmPotentialSolve<spaceDim>::ScalarVector rhs1D  = Kokkos::subview (rhs,              0, Kokkos::ALL() );
    
    typename LowRmPotentialSolve<spaceDim>::ScalarVector expectedRHS ("expectedRHS", numNodes);
    matrix.Apply(soln1D,expectedRHS);
    
    out << "\n**comparing expected RHS to actual.**\n";
    bool mySuccess = true;
    testFloatingEquality(expectedRHS, rhs1D, tol, out, mySuccess);
    
    success = success && mySuccess;
    if (!mySuccess)
    {
      out << "\n*********** Details of failed Ax = b consistency check **************\n";
      out << "\n***********                   Matrix A                 **************\n";
      MatrixIO::writeSparseMatlabMatrix(out, matrix);
      out << "\n***********                   soln x                   **************\n";
      MatrixIO::writeDenseMatlabVector(out, soln1D);
      out << "\n***********                   RHS b                    **************\n";
      MatrixIO::writeDenseMatlabVector(out, rhs1D);
      out << "\n***********                   Product Ax               **************\n";
      MatrixIO::writeDenseMatlabVector(out, expectedRHS);
    }
  }
  
  template<int spaceDim>
  void testCircuitLumping(Teuchos::FancyOStream &out, bool &success)
  {
    /*
     Here, we determine the exact discrete value we expect for a given mesh.
     
     The way this works: we basically reason from the 1D solution, and use the fact that we are
     nodally exact in 1D.  Thus we know at both ends of an 1D element what we expect the values 
     to be.  The difference, divided by the element width h, is the gradient.  The square of this
     times the element width is the integral contribution from that element.
     
     I believe all of this should carry over directly to 2D and 3D, since all our solutions only
     vary in x.
     */
    
    
    auto exactSolutions = getExactPolynomialSolutions();
    
    for (auto exactSolution : exactSolutions)
    {
      auto solnExpr        = exactSolution.exactSolution;
      
      std::vector<int> meshWidths = {1,2,3};
      
      for (int meshWidth : meshWidths)
      {
        auto mesh = getBoxMesh(spaceDim, meshWidth);
        LowRmPotentialSolve<spaceDim> solver = getLowRmPotentialSolveExample<spaceDim>(mesh);
        
        solver.initialize();
        
        int numRHSes = 1;
        int numNodes = mesh->nverts();
        typename LowRmPotentialSolve<spaceDim>::ScalarMultiVector expectedSolution("expected solution",numRHSes,numNodes);
        
        auto coords            = mesh->coords();
        auto expectedSoln_1D_subview = Kokkos::subview (expectedSolution, 0, Kokkos::ALL ());
        evaluateNodalExpression(solnExpr, spaceDim, coords, expectedSoln_1D_subview);
        
        // determine expected_K11 value
        double expected_K11 = getExpectedK11(meshWidth, solnExpr);
        
        // copy the expected solution into LHS vector, skipping the actual assembly and solve steps...
        Kokkos::deep_copy(solver.getLHS(), expectedSolution);
        
        // to avoid a warning, call getJouleHeating() first:
        {
          ScalarVector emptyVector1;
          ScalarVector emptyVector2;
          bool warnAboutEmptyVector = false;
          solver.determineJouleHeating(emptyVector1, emptyVector2, 0.0, 0.0, warnAboutEmptyVector);
        }
        
        Scalar actual_K11 = solver.getK11();
        double tol = 1e-14;
        TEST_FLOATING_EQUALITY(expected_K11, actual_K11, tol);
      }
    }
  }
  
  void testMatrixHasNoZeroRows(CrsMatrix matrix, Teuchos::FancyOStream &out, bool &success)
  {
    auto rowMap = matrix.rowMap();
    double floor = 1e-15;
    int numZeroRows = 0;

    int rowCount = rowMap.size() - 1;
    
    Kokkos::parallel_reduce("Determine zero row count", rowCount, LAMBDA_EXPRESSION(int row, int &localZeroCount)
    {
      auto entryCount = matrix.rowMap()(row+1) - matrix.rowMap()(row);
      auto entryOffset = matrix.rowMap()(row);
      bool nonZeroFound = false;
      for (int entryOrdinal=0; entryOrdinal<int(entryCount); entryOrdinal++)
      {
        if (std::abs(matrix.entries()(entryOffset + entryOrdinal)) > floor)
        {
          nonZeroFound = true;
          break;
        }
      }
      if (!nonZeroFound)
      {
        localZeroCount++;
      }
    }, numZeroRows);
    
    // TODO: find out why the below doesn't compile (issues with const status of numZeroRows)
    //       (this should do precisely what the reduce above does; I just want to understand how I could do it something like below)
//    Kokkos::parallel_for("check for zero rows", matrix.rowMap().size() - 1, LAMBDA_EXPRESSION(int row)
//     {
//       auto entryCount = matrix.rowMap()(row+1) - matrix.rowMap()(row);
//       auto entryOffset = matrix.rowMap()(row);
//       bool nonZeroFound = false;
//       for (int entryOrdinal=0; entryOrdinal<int(entryCount); entryOrdinal++)
//       {
//         if (std::abs(matrix.entries()(entryOffset + entryOrdinal)) > floor)
//         {
//           nonZeroFound = true;
//           break;
//         }
//       }
//       if (!nonZeroFound)
//       {
//         Kokkos::atomic_add(&numZeroRows,1);
//       }
//     });
    TEST_EQUALITY(0, numZeroRows);
    if (numZeroRows > 0)
    {
      out << "matrix with numZeroRows > 0:\n";
      MatrixIO::writeSparseMatlabMatrix(out, matrix);
    }
  }
  
  template<int spaceDim>
  void testPolynomialExactSolution(int meshWidth, MatrixSolver matrixSolver, Teuchos::FancyOStream &out, bool &success, double tol=1e-12)
  {
    auto mesh = getBoxMesh(spaceDim, meshWidth);
    LowRmPotentialSolve<spaceDim> solver = getLowRmPotentialSolveExample<spaceDim>(mesh);
    auto exactSolutions = getExactPolynomialSolutions();
    
    for (auto exactSolution : exactSolutions)
    {
      auto solnExpr        = exactSolution.exactSolution;
      auto forcingExpr     = exactSolution.forcingFunction;
      int quadratureDegree = exactSolution.forcingQuadratureDegree;
      
      solver.setForcingFunctionExpr(forcingExpr, quadratureDegree);
      
      out << "*******   testing exact solution " << solnExpr << "  ********\n";
      
      // boundary conditions
      auto boundaryNodes = getBoundaryNodes(mesh);
      solver.setBC(solnExpr, boundaryNodes);
      
      solver.initialize();
      solver.assemble();
      testMatrixHasNoZeroRows(solver.getMatrix(), out, success);
      
      int numRHSes = 1;
      int numNodes = mesh->nverts();
      typename LowRmPotentialSolve<spaceDim>::ScalarMultiVector expectedSolution("expected solution",numRHSes,numNodes);
      
      auto coords            = mesh->coords();
      auto expectedSoln_1D_subview = Kokkos::subview (expectedSolution, 0, Kokkos::ALL ());
      evaluateNodalExpression(solnExpr, spaceDim, coords, expectedSoln_1D_subview);
      
      testAssembledSystemMatchesSolution(solver, expectedSolution, out, success, tol);

      if (matrixSolver != NONE)
      {
        auto A = solver.getMatrix();
        auto b = solver.getRHS();
        auto x = solver.getLHS();
        
        int result = 0;
        if (matrixSolver == VIENNACL)
        {
#ifdef HAVE_VIENNA_CL
          // clear x
          Kokkos::deep_copy(x, 0);
          ViennaLinearProblem problem(A, x, b);
          problem.setTolerance(1e-15);
          result = problem.solve();
          
          out << "ViennaCL solved in " << problem.getIterationsTaken() << " iterations, with residual " << problem.getResidual() << std::endl;
          
#else
          out << "test was run with matrixSolver = VIENNACL, but HAVE_VIENNA_CL is not defined!\n";
          success = false;
#endif
        }
        else if (matrixSolver == AMGX)
        {
#ifdef HAVE_AMGX
          // clear x
          Kokkos::deep_copy(x, 0);
          double cgTol = 1e-15;
          int maxIters = 10000;
          AmgXLinearProblem problem(A, x, b, AmgXLinearProblem::configurationString(AmgXLinearProblem::EAF,cgTol,maxIters));
          result = problem.solve();
#else
          out << "test was run with matrixSolver = AMGX, but HAVE_AMGX is not defined!\n";
          success = false;
#endif
        }
        TEST_EQUALITY(0, result);
        
        out << "\n**comparing solutions**\n";
        testFloatingEquality(expectedSolution, x, tol, out, success);
        
        //      { // DEBUGGING
        //        std::cout << "A:\n";
        //        MatrixIO::writeSparseMatlabMatrix(std::cout, solver.getMatrix());
        //        std::cout << "b:\n";
        //        MatrixIO::writeDenseMatlabVector(std::cout, Kokkos::subview(b, 0, Kokkos::ALL()));
        //        std::cout << "x:\n";
        //        MatrixIO::writeDenseMatlabVector(std::cout, Kokkos::subview(x, 0, Kokkos::ALL()));
        //
        //        ScalarVector expectedSolution_1D = Kokkos::subview(expectedSolution, 0, Kokkos::ALL() );
        //        MatrixIO::writeDenseMatlabVector(std::cout, expectedSolution_1D);
        //      }
      }
    }
    
    //    auto matrix = solver.getMatrix();
    //    auto rhs    = solver.getRHS();
    //    std::cout << "matrix:\n";
    //    MatrixIO::writeSparseMatlabMatrix(std::cout, matrix);
    //
    //    std::cout << "rhs:\n";
    //    MatrixIO::writeDenseMatlabMatrix(std::cout, rhs);
    //
    //    std::cout << "expected LHS:\n";
    //    MatrixIO::writeDenseMatlabMatrix(std::cout, expectedSolution);
    //    double tol = 1e-15;
    //    testFloatingEquality<Scalar, Ordinal, RowMapEntryType, Layout, MemSpace>(matrix,matrixOut,tol,out,success);
  }

  TEUCHOS_UNIT_TEST( LowRmPotentialSolve, CircuitLumping_1D )
  {
    const int spaceDim = 1;
    testCircuitLumping<spaceDim>(out, success);
  }
  
  TEUCHOS_UNIT_TEST( LowRmPotentialSolve, CircuitLumping_2D )
  {
    const int spaceDim = 2;
    testCircuitLumping<spaceDim>(out, success);
  }
  
  TEUCHOS_UNIT_TEST( LowRmPotentialSolve, CircuitLumping_3D )
  {
    const int spaceDim = 3;
    testCircuitLumping<spaceDim>(out, success);
  }
  
  TEUCHOS_UNIT_TEST( LowRmPotentialSolve, Initialize_1D )
  {
    /*
     Test the initialization of the matrix, LHS, and RHS:
     - in 1D, the number of degrees of freedom should be equal to the meshWidth plus 1
     - in 1D, the number of nonzeros in the matrix should be 3 * numDofs - 2.
     - maybe also test that rowMap is what we expect?
     */
    
    const int spaceDim = 1;
    const int meshWidth = 2;
    auto mesh = getBoxMesh(spaceDim, meshWidth);
    
    LowRmPotentialSolve<spaceDim> solver = getLowRmPotentialSolveExample<spaceDim>(mesh);
    
    solver.initialize();
    
    const int expectedDofCount = meshWidth + 1;
    const int expectedNNZ      = expectedDofCount * 3 - 2;
    
    TEST_EQUALITY(expectedDofCount, solver.getLHS().size());
    TEST_EQUALITY(expectedDofCount, solver.getRHS().size());
    
    TEST_EQUALITY(expectedDofCount+1, solver.getMatrix().rowMap().size());
    TEST_EQUALITY(expectedNNZ, solver.getMatrix().columnIndices().size());
    
    // test that the row map and column indices are what we expect
    // assuming that that dofs are numbered from left to right or right to left,
    // the boundary dofs will correspond to first and last rows.  This means that
    // we expect to have two entries in first and last rows, and three in every other
    CrsMatrix::RowMapVector  expectedRowMap("expected row map", expectedDofCount + 1);
    CrsMatrix::OrdinalVector expectedColumnIndices("expected column indices", expectedNNZ);

    Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,expectedDofCount+1), LAMBDA_EXPRESSION(int rowMapEntryOrdinal)
    {
     int rowStart, rowEnd;
     int rowEntries[3] = {0, 0, 0}; // the most we'll need; initialized to avoid compiler warnings
     if (rowMapEntryOrdinal == 0)
     {
       // first row; we expect 2 entries
       rowStart = 0;
       rowEnd   = 2;
       rowEntries[0] = 0;
       rowEntries[1] = 1;
     }
     else if (rowMapEntryOrdinal == expectedDofCount)
     {
       rowStart = expectedNNZ;
       rowEnd   = expectedNNZ;
       // no need to initialize rowEntries
     }
     else
     {
       rowStart = (rowMapEntryOrdinal - 1) * 3 + 2;
       if (rowMapEntryOrdinal == expectedDofCount - 1)
       {
         // last row; second-to-last entry
         rowEnd = rowStart + 2;
         rowEntries[0] = rowMapEntryOrdinal - 1;
         rowEntries[1] = rowMapEntryOrdinal;
       }
       else
       {
         rowEnd = rowStart + 3;
         rowEntries[0] = rowMapEntryOrdinal - 1;
         rowEntries[1] = rowMapEntryOrdinal;
         rowEntries[2] = rowMapEntryOrdinal + 1;
       }
     }
     
     expectedRowMap(rowMapEntryOrdinal) = rowStart;
     for (int entryOrdinal=rowStart; entryOrdinal<rowEnd; entryOrdinal++)
     {
       expectedColumnIndices(entryOrdinal) = rowEntries[entryOrdinal-rowStart];
     }
    }, "fill expected row map and column indices");
    testEquality<RowMapEntryType>(expectedRowMap, solver.getMatrix().rowMap(), out, success);
    testEquality<Ordinal> (expectedColumnIndices, solver.getMatrix().columnIndices(), out, success);
  }
  
  TEUCHOS_UNIT_TEST( LowRmPotentialSolve, Initialize_2D )
  {
    /*
     Test the initialization of the matrix, LHS, and RHS:
     - in 2D, the number of degrees of freedom should be equal to (meshWidth+1)^2
     - in 2D, there are three types of dofs:
       - middle dofs.  Given the way we divide into boxes, these should have 6 other dofs that they talk to (7 total)
       - boundary edge dofs.  These talk to 4 others (5 total)
       - boundary corner dofs.  These talk to 2 or 3 others, depending on the corner (3 or 4 total)
     - The number of middle dofs is (meshWidth - 1)^2
     - The number of boundary edge dofs is (meshWidth - 1)*4
     - The number of corner dofs is 4.
     - Middle dofs contribute 7*(meshWidth-1)^2 nonzeros
     - Boundary edge dofs contribute 5*(meshWith-1)*4 nonzeros
     - Corner dofs contribute 14 nonzeros.
     - in 2D, the number of nonzeros in the matrix should be 14 + 20*(meshWidth-1) + (meshWidth-1)*(meshWidth-1)
     - maybe also test that rowMap is what we expect?
     */
    
    const int spaceDim = 2;
    const int meshWidth = 2;
    auto mesh = getBoxMesh(spaceDim, meshWidth);
    
    LowRmPotentialSolve<spaceDim> solver = getLowRmPotentialSolveExample<spaceDim>(mesh);
    
    solver.initialize();
    
    const int expectedDofCount = (meshWidth + 1)*(meshWidth+1);
    const int expectedNNZ      = 14 + 20*(meshWidth-1) + 7*(meshWidth-1)*(meshWidth-1);
    
    TEST_EQUALITY(expectedDofCount, solver.getLHS().size());
    TEST_EQUALITY(expectedDofCount, solver.getRHS().size());
    
    TEST_EQUALITY(expectedDofCount+1, solver.getMatrix().rowMap().size());
    TEST_EQUALITY(expectedNNZ, solver.getMatrix().columnIndices().size());
    
    // test that the row map and column indices are what we expect
    // assuming that that dofs are numbered from left to right or right to left,
    // the boundary dofs will correspond to first and last rows.  This means that
    // we expect to have two entries in first and last rows, and three in every other
    
    // the below is the 1D code; will need some thought as to what we expect in 2D
  //CrsMatrix::RowMapVector  expectedRowMap("expected row map", expectedDofCount + 1);
  //CrsMatrix::OrdinalVector expectedColumnIndices("expected column indices", expectedNNZ);
  //
  //Kokkos::parallel_for("fill expected row map and column indices", expectedDofCount+1, LAMBDA_EXPRESSION(int rowMapEntryOrdinal)
  //                     {
  //                       int rowStart, rowEnd;
  //                       int rowEntries[3]; // the most we'll need
  //                       if (rowMapEntryOrdinal == 0)
  //                       {
  //                         // first row; we expect 2 entries
  //                         rowStart = 0;
  //                         rowEnd   = 2;
  //                         rowEntries[0] = 0;
  //                         rowEntries[1] = 1;
  //                       }
  //                       else if (rowMapEntryOrdinal == expectedDofCount)
  //                       {
  //                         rowStart = expectedNNZ;
  //                         rowEnd   = expectedNNZ;
  //                         // no need to initialize rowEntries
  //                       }
  //                       else
  //                       {
  //                         rowStart = (rowMapEntryOrdinal - 1) * 3 + 2;
  //                         if (rowMapEntryOrdinal == expectedDofCount - 1)
  //                         {
  //                           // last row; second-to-last entry
  //                           rowEnd = rowStart + 2;
  //                           rowEntries[0] = rowMapEntryOrdinal - 1;
  //                           rowEntries[1] = rowMapEntryOrdinal;
  //                         }
  //                         else
  //                         {
  //                           rowEnd = rowStart + 3;
  //                           rowEntries[0] = rowMapEntryOrdinal - 1;
  //                           rowEntries[1] = rowMapEntryOrdinal;
  //                           rowEntries[2] = rowMapEntryOrdinal + 1;
  //                         }
  //                       }
  //                       
  //                       expectedRowMap(rowMapEntryOrdinal) = rowStart;
  //                       for (int entryOrdinal=rowStart; entryOrdinal<rowEnd; entryOrdinal++)
  //                       {
  //                         expectedColumnIndices(entryOrdinal) = rowEntries[entryOrdinal-rowStart];
  //                       }
  //                     });
  //testEquality<RowMapEntryType>(expectedRowMap, solver.getMatrix()->rowMap(), out, success);
  //testEquality<Ordinal> (expectedColumnIndices, solver.getMatrix()->columnIndices(), out, success);
  }
  
  TEUCHOS_UNIT_TEST( LowRmPotentialSolve, MFoldSymmetry_3D )
  {
    // in 1D, we think of slicing a 1D wire into m equal lengths.  We solve the problem on one such segment,
    // using appropriately modified BCs.  We expect the K11 value we get out to be the same regardless of m.
    
    // The above is for "series symmetry."  TODO: test parallel symmetry.
    
    const int spaceDim = 3;
    std::vector<int> mSeriesValues = {1,2,4,8};
    
    int mParallel = 1;
    for (int mSeries : mSeriesValues)
    {
      int meshWidth = 2;
      double x_scaling = 1.0 / double(mSeries);
      auto mesh = getBoxMesh(spaceDim, meshWidth, x_scaling, 1.0, 1.0); // scale in x, but keep y,z range as [0,1]

      LowRmPotentialSolve<spaceDim> solver = getLowRmPotentialSolveExample<spaceDim>(mesh);
      
      solver.initialize();
      solver.setUseMFoldSymmetry(mSeries, mParallel);
      
      // TODO: work out a way to label the side sets appropriately in the mesh (meshIO), so that we can use solver.setPorts(<#const MeshIO &meshIO#>, <#std::string &inputNodeSetName#>, <#std::string &outputNodeSetName#>)
      // instead of the below, which has to modify the BC imposition in much the same way as the
      Omega_h::LOs x0_ordinals = getBoundaryNodes_x0(mesh);
      Omega_h::LOs x1_ordinals = getBoundaryNodes_x1(mesh);
      
      Omega_h::Write<LO> bcOrdinals(x0_ordinals.size() + x1_ordinals.size());
      Omega_h::Write<double> bcValues(bcOrdinals.size());
      
      Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,x0_ordinals.size()), LAMBDA_EXPRESSION(int x0_ordinal)
      {
        bcOrdinals[x0_ordinal] = x0_ordinals[x0_ordinal];
        bcValues  [x0_ordinal] = 1.0 - 1.0/double(mSeries);
      }, "Dirichlet BCs -- phi=xMin");
      
      auto offset = x0_ordinals.size();
      Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,x1_ordinals.size()), LAMBDA_EXPRESSION(int x1_ordinal)
      {
        bcOrdinals[offset + x1_ordinal] = x1_ordinals[x1_ordinal];
        bcValues  [offset + x1_ordinal] = 1.0;
      }, "Dirichlet BCs -- phi=xMax");
      
      solver.setBC(bcOrdinals, bcValues);
      
      double expectedK11 = getExpectedK11(meshWidth*mSeries*mParallel, "x"); // x is the exact solution, given our BCs, on [0,1]
      
      int numNodes = mesh->nverts();
      int numRHSes = 1;
      typename LowRmPotentialSolve<spaceDim>::ScalarMultiVector expectedSolution("expected solution",numRHSes,numNodes);
      
      auto coords            = mesh->coords();
      auto expectedSoln_1D_subview = Kokkos::subview (expectedSolution, 0, Kokkos::ALL ());
      
      std::string solnExpr;
      {
        // given our [0,1/m] domain, and the BCs we are imposing on it, the exact solution here is
        // x + 1-1/m
        double offset2 = 1.0 - 1.0 / double(mSeries);
        std::ostringstream solnStream;
        solnStream << "x + " << offset2;
        solnExpr = solnStream.str();
      }
      evaluateNodalExpression(solnExpr, spaceDim, coords, expectedSoln_1D_subview);
      
      // copy the expected solution into LHS vector, skipping the actual assembly and solve steps...
      Kokkos::deep_copy(solver.getLHS(), expectedSolution);
      
      // to avoid a warning, call getJouleHeating() first:
      {
        ScalarVector emptyVector1;
        ScalarVector emptyVector2;
        bool warnAboutEmptyVector = false;
        solver.determineJouleHeating(emptyVector1, emptyVector2,.0, 0.0, warnAboutEmptyVector);
      }
      
      Scalar actualK11 = solver.getK11();
      
      double tol = 1e-13;
      TEST_FLOATING_EQUALITY(expectedK11, actualK11, tol);
    }
  }
  
  
  TEUCHOS_UNIT_TEST( LowRmPotentialSolve, PolynomialExactSolution_1D_2_Wide )
  {
    const int spaceDim = 1;
    int meshWidth = 2;
    auto mesh = getBoxMesh(spaceDim, meshWidth);
    
    MatrixSolver matrixSolver = NONE;
    testPolynomialExactSolution<spaceDim>(meshWidth, matrixSolver, out, success);
  }
  
  TEUCHOS_UNIT_TEST( LowRmPotentialSolve, PolynomialExactSolution_1D )
  {
    const int spaceDim = 1;
    int meshWidth = 3;
    MatrixSolver matrixSolver = NONE;
    testPolynomialExactSolution<spaceDim>(meshWidth, matrixSolver, out, success);
  }
  
  TEUCHOS_UNIT_TEST( LowRmPotentialSolve, PolynomialExactSolution_1D_WideMesh )
  {
    const int spaceDim = 1;
    int meshWidth = 16;
    MatrixSolver matrixSolver = NONE;
    testPolynomialExactSolution<spaceDim>(meshWidth, matrixSolver, out, success);
  }
  
  TEUCHOS_UNIT_TEST( LowRmPotentialSolve, PolynomialExactSolution_2D )
  {
    const int spaceDim = 2;
    int meshWidth = 2;
    MatrixSolver matrixSolver = NONE;
    testPolynomialExactSolution<spaceDim>(meshWidth, matrixSolver, out, success);
  }
  
//  TEUCHOS_UNIT_TEST( LowRmPotentialSolve, PolynomialExactSolution_2D_WideMesh8 )
//  {
//    const int spaceDim = 2;
//    int meshWidth = 8;
//    MatrixSolver matrixSolver = NONE;
//    testPolynomialExactSolution<spaceDim>(meshWidth, matrixSolver, out, success);
//  }

  TEUCHOS_UNIT_TEST( LowRmPotentialSolve, PolynomialExactSolution_2D_WideMesh11 )
  {
    const int spaceDim = 2;
    int meshWidth = 11;
    MatrixSolver matrixSolver = NONE;
    testPolynomialExactSolution<spaceDim>(meshWidth, matrixSolver, out, success);
  }
  
//  TEUCHOS_UNIT_TEST( LowRmPotentialSolve, PolynomialExactSolution_2D_WideMesh12 )
//  {
//    const int spaceDim = 2;
//    int meshWidth = 12;
//    MatrixSolver matrixSolver = NONE;
//    testPolynomialExactSolution<spaceDim>(meshWidth, matrixSolver, out, success);
//  }
//  
//  TEUCHOS_UNIT_TEST( LowRmPotentialSolve, PolynomialExactSolution_2D_WideMesh16 )
//  {
//    const int spaceDim = 2;
//    int meshWidth = 16;
//    MatrixSolver matrixSolver = NONE;
//    testPolynomialExactSolution<spaceDim>(meshWidth, matrixSolver, out, success);
//  }
  
  TEUCHOS_UNIT_TEST( LowRmPotentialSolve, PolynomialExactSolution_3D )
  {
    const int spaceDim = 3;
    int meshWidth = 2;
    MatrixSolver matrixSolver = NONE;
    testPolynomialExactSolution<spaceDim>(meshWidth, matrixSolver, out, success);
  }
  
//  TEUCHOS_UNIT_TEST( LowRmPotentialSolve, PolynomialExactSolution_3D_WideMesh16 )
//  {
//    const int spaceDim = 3;
//    int meshWidth = 16;
//    MatrixSolver matrixSolver = NONE;
//    testPolynomialExactSolution<spaceDim>(meshWidth, matrixSolver, out, success);
//  }

#ifdef HAVE_VIENNA_CL
  TEUCHOS_UNIT_TEST( LowRmPotentialSolve, PolynomialExactSolution_1D_ViennaCL )
  {
    const int spaceDim = 1;
    int meshWidth = 3;
    MatrixSolver matrixSolver = VIENNACL;
    testPolynomialExactSolution<spaceDim>(meshWidth, matrixSolver, out, success);
  }
  
  TEUCHOS_UNIT_TEST( LowRmPotentialSolve, PolynomialExactSolution_2D_ViennaCL )
  {
    const int spaceDim = 2;
    int meshWidth = 2;
    MatrixSolver matrixSolver = VIENNACL;
    testPolynomialExactSolution<spaceDim>(meshWidth, matrixSolver, out, success);
  }
  
  TEUCHOS_UNIT_TEST( LowRmPotentialSolve, PolynomialExactSolution_3D_ViennaCL )
  {
    const int spaceDim = 3;
    int meshWidth = 2;
    MatrixSolver matrixSolver = VIENNACL;
    testPolynomialExactSolution<spaceDim>(meshWidth, matrixSolver, out, success);
  }
  
  TEUCHOS_UNIT_TEST( LowRmPotentialSolve, PolynomialExactSolution_3D_Width8_ViennaCL )
  {
    const int spaceDim = 3;
    int meshWidth = 8;
    MatrixSolver matrixSolver = VIENNACL;
    double tol=1e-11;
    testPolynomialExactSolution<spaceDim>(meshWidth, matrixSolver, out, success, tol);
  }
#endif
  
#ifdef HAVE_AMGX
  TEUCHOS_UNIT_TEST( LowRmPotentialSolve, PolynomialExactSolution_1D_AmgX )
  {
    const int spaceDim = 1;
    int meshWidth = 3;
    MatrixSolver matrixSolver = AMGX;
    testPolynomialExactSolution<spaceDim>(meshWidth, matrixSolver, out, success);
  }
  
  TEUCHOS_UNIT_TEST( LowRmPotentialSolve, PolynomialExactSolution_2D_AmgX )
  {
    const int spaceDim = 2;
    int meshWidth = 2;
    MatrixSolver matrixSolver = AMGX;
    testPolynomialExactSolution<spaceDim>(meshWidth, matrixSolver, out, success);
  }
  
  TEUCHOS_UNIT_TEST( LowRmPotentialSolve, PolynomialExactSolution_3D_AmgX )
  {
    const int spaceDim = 3;
    int meshWidth = 2;
    MatrixSolver matrixSolver = AMGX;
    testPolynomialExactSolution<spaceDim>(meshWidth, matrixSolver, out, success);
  }

  // TODO: figure out why this test does not pass on white
//  TEUCHOS_UNIT_TEST( LowRmPotentialSolve, PolynomialExactSolution_3D_Width8_AmgX )
//  {
//    const int spaceDim = 3;
//    int meshWidth = 8;
//    MatrixSolver matrixSolver = AMGX;
//    double tol = 1e-11;
//    testPolynomialExactSolution<spaceDim>(meshWidth, matrixSolver, out, success, tol);
//  }
#endif
  
  TEUCHOS_UNIT_TEST( LowRmPotentialSolve, RHS_UnitForcing_2D )
  {
    /*
     A simple sanity check on the RHS assembly in 2D.  If we have a 1-wide mesh and unit forcing,
     the element-wise integrals of (-f,v) for each basis function v will be -1/6.  (Think of the volume of
     a tetrahedron with a right triangle as its base.)
     
     Now, two vertices will have two such contributions, while the other two will have only one.  So in some
     order we expect (-1/6, -1/6, -1/3, -1/3) as the values on the RHS.  It turns out that nodes are numbered such
     that the two that are on the diagonal are even.
     
     */
    const int spaceDim = 2;
    int meshWidth = 1;
    double scaling = 0.5;
    auto mesh = getBoxMesh(spaceDim, meshWidth, 0.5);
    LowRmPotentialSolve<spaceDim> solver = getLowRmPotentialSolveExample<spaceDim>(mesh);
    
    solver.setForcingFunctionExpr("1", 0);
    
    solver.initialize();
    solver.assemble();
    
    int numRHSes = 1;
    int numNodes = mesh->nverts();
    LowRmPotentialSolve<spaceDim>::ScalarMultiVector expectedRHS("b",numRHSes,numNodes);
    
    auto coords            = mesh->coords();
    Kokkos::parallel_for(Kokkos::RangePolicy<int>(0, numNodes), LAMBDA_EXPRESSION(int nodeNumber)
    {
     if (nodeNumber % 2 == 0)
       expectedRHS(0,nodeNumber) = -1.0 / 3.0 * scaling * scaling;
     else
       expectedRHS(0,nodeNumber) = -1.0 / 6.0 * scaling * scaling;
    }, "initialize solution");
    
    auto rhs    = solver.getRHS();
    
//    std::cout << "rhs:\n";
//    MatrixIO::writeDenseMatlabMatrix(std::cout, rhs);
    
    double tol = 1e-15;
    testFloatingEquality(expectedRHS,rhs,tol,out,success);
  }
} // namespace
