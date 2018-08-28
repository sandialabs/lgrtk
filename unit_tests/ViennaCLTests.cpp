#include "Teuchos_UnitTestHarness.hpp"

#include <ViennaSparseLinearProblem.hpp>

#include <vector>

#include "LGRTestHelpers.hpp"

using namespace lgr;
using namespace std;

namespace {
  typedef double                                                                       Scalar;
  typedef int                                                                          Ordinal;
  typedef Kokkos::LayoutLeft                                                           Layout;
  typedef Kokkos::DefaultExecutionSpace                                                MemSpace;
  typedef typename Kokkos::ViewTraits<Ordinal*, Layout, MemSpace, void >::size_type    SizeType;
  typedef ViennaSparseLinearProblem<Scalar, Ordinal, Layout, MemSpace>                 LinearProblem;
  
  typedef CrsMatrix<Scalar, Ordinal, SizeType, Layout, MemSpace>          CrsMatrix;
//  typedef MatrixIO <Scalar, Ordinal, SizeType, Layout, MemSpace>          MatrixIO;
  
  typedef Kokkos::View<Ordinal*,  Layout, MemSpace> OrdinalVector;
  typedef Kokkos::View<Scalar* ,  Layout, MemSpace> ScalarVector;
  typedef Kokkos::View<SizeType*, Layout, MemSpace> SizeTypeVector;
  
  CrsMatrix sampleCrsMatrix(int numRows)
  {
    using namespace std;
    const int VALUES_PER_ROW = 3;
    int nnz = VALUES_PER_ROW * numRows - 2; // first, last row get truncated: hence the -2
    
    SizeTypeVector rowMap      ( "rowMap",        numRows+1 );
    OrdinalVector columnIndices( "columnIndices", nnz       );
    ScalarVector  entries      ( "entries",       nnz       );
    
    int minColumnIndex = 0;
    int maxColumnIndex = numRows-1;
    
    Kokkos::parallel_for("initialize sample CRS matrix", numRows, LAMBDA_EXPRESSION(int row)
                         {
                           const Scalar values[VALUES_PER_ROW] = {1.0, 2.0, 1.0}; // for each row
                           const int colOrdinalOffsets[VALUES_PER_ROW] = {-1, 0, 1}; // we don't allow negative indices; if we'd store one we truncate
                           int entryOffset = (row == 0) ? 0 : 2 + (row-1) * VALUES_PER_ROW;
                           //      cout << "row " << row << ", entryOffset = " << entryOffset << endl;
                           for (int colOrdinal=0; colOrdinal<VALUES_PER_ROW; colOrdinal++)
                           {
                             int columnIndex = row + colOrdinalOffsets[colOrdinal];
                             if ((columnIndex < minColumnIndex) || (columnIndex > maxColumnIndex)) continue; // truncate to columns
                             columnIndices(entryOffset) = columnIndex;
                             Scalar value = values[colOrdinal];
                             entries(entryOffset) = value;
                             //        cout << "A[" << row << "][" << columnIndex <<"] = " << value << endl;
                             //        cout << "columnIndices(" << entryOffset << ") = " << columnIndex << endl;
                             
                             entryOffset++;
                           }
                           rowMap(row+1) = entryOffset;
                         });
    
    return CrsMatrix(rowMap, columnIndices, entries);
  }
  
  ScalarVector sampleLHS(int numRows)
  {
    ScalarVector x("x",numRows);
    // just do rowNumber / 2
    
    Kokkos::parallel_for("initialize sample CRS matrix", numRows, LAMBDA_EXPRESSION(int row)
                         {
                           x(row) = Scalar(row) / 2.0;
                         });
    return x;
  }
  
  ScalarVector sampleRHS(int numRows)
  {
    ScalarVector b("b",numRows);
    // built to be consistent with A = sampleCrsMatrix and x = sampleLHS, so that A * x = b
    const int VALUES_PER_ROW = 3;
    vector<Scalar> A_values = {1.0, 2.0, 1.0}; // for each row
    vector<int> colOrdinalOffsets = {-1, 0, 1}; // we don't allow negative indices; if we'd store one we truncate
    int minColumnIndex = 0;
    int maxColumnIndex = numRows-1;
    
    Kokkos::parallel_for("initialize sample RHS", numRows, LAMBDA_EXPRESSION(int row)
                         {
                           const Scalar values[VALUES_PER_ROW] = {1.0, 2.0, 1.0}; // for each row
                           const int colOrdinalOffsets[VALUES_PER_ROW] = {-1, 0, 1}; // we don't allow negative indices; if we'd store one we truncate
                           //      cout << "row " << row << ", entryOffset = " << entryOffset << endl;
                           Scalar rowValue = 0.0; // in RHS
                           for (int colOrdinal=0; colOrdinal<VALUES_PER_ROW; colOrdinal++)
                           {
                             int columnIndex = row + colOrdinalOffsets[colOrdinal];
                             if ((columnIndex < minColumnIndex) || (columnIndex > maxColumnIndex)) continue; // truncate to columns
                             
                             Scalar A_value = values[colOrdinal];
                             Scalar xValue = Scalar(columnIndex) / 2.0;
                             rowValue += xValue * A_value;
                           }
                           b(row) = rowValue;
                         });
    
    return b;
  }
  
  TEUCHOS_UNIT_TEST( ViennaCL, SolveSampleSystem )
  {
    const int numRows = 25;
    ScalarVector xActual("x",numRows);
    ScalarVector xExpected = sampleLHS(numRows);
    ScalarVector b = sampleRHS(numRows);
    CrsMatrix A = sampleCrsMatrix(numRows);
    
    LinearProblem problem(A, xActual, b);
    int result = problem.solve();
    
    TEST_EQUALITY(0, result);
    
    double tol = 1e-12;
    testFloatingEquality(xExpected, xActual, tol, out, success);
  }
  
  TEUCHOS_UNIT_TEST( ViennaCL, SolveSampleSystem2 )
  {
    /*
     Sample System 2 is really very simple:
         [ 1  0   ]      [   0   ]       [  0   ]
     A = [ 0  4  0]  b = [  1/12 ]   x = [ 1/48 ]
         [    0  1]      [  1/6  ]       [ 1/6  ]
     */
    
    int numRows = 3;
    ScalarVector xActual("x",numRows);
    ScalarVector xExpected("x expected", numRows);
    ScalarVector b("b", numRows);
    
    int nnz = 7; // 2 for first, 3 for second, 2 for third
    
    SizeTypeVector rowMap      ( "rowMap",        numRows+1 );
    OrdinalVector columnIndices( "columnIndices", nnz       );
    ScalarVector  entries      ( "entries",       nnz       );
    
    Kokkos::parallel_for("initialize sample problem 2", numRows, LAMBDA_EXPRESSION(int row)
    {
      if (row == 0)
      {
        rowMap(0) = 0;
        rowMap(1) = 2;
        
        columnIndices(0) = 0;
        columnIndices(1) = 1;
        entries(0) = 1.0;
        entries(1) = 0.0;
        
        xExpected(0) = 0.0;
        b(0) = 0.0;
      }
      else if (row == 1)
      {
        rowMap(1) = 2;
        rowMap(2) = 5;
        
        columnIndices(2) = 0;
        columnIndices(3) = 1;
        columnIndices(4) = 2;
        entries(2) =  0.0;
        entries(3) =  4.0;
        entries(4) =  0.0;
        
        xExpected(1) = 1.0 / 48.0;
        b(1) =  1.0 / 12.0;
      }
      else if (row == 2)
      {
        rowMap(2) = 5;
        rowMap(3) = nnz;
        
        columnIndices(5) = 1;
        columnIndices(6) = 2;
        entries(5) = 0.0;
        entries(6) = 1.0;
        
        xExpected(2) = 1.0 / 6.0;
        b(2) = 1.0 / 6.0;
      }
    });
    
    CrsMatrix A(rowMap, columnIndices, entries);
    LinearProblem problem(A, xActual, b);
    int result = problem.solve();
    
    TEST_EQUALITY(0, result);
    
    double tol = 1e-12;
    testFloatingEquality(xExpected, xActual, tol, out, success);
  }

//  TEUCHOS_UNIT_TEST( ViennaCL, SolveSampleSystem3 )
//  {
//    /*
//     Sample System 3 is a lot like Sample System 2, but with no zeros stored in A
//         [ 1      ]      [   0   ]       [  0   ]
//     A = [    4   ]  b = [  1/12 ]   x = [ 1/48 ]
//         [       1]      [  1/6  ]       [ 1/6  ]
//     */
//    
//    int numRows = 3;
//    ScalarVector xActual("x",numRows);
//    ScalarVector xExpected("x expected", numRows);
//    ScalarVector b("b", numRows);
//    
//    int nnz = 3; // just the diagonal gets stored
//    
//    SizeTypeVector rowMap      ( "rowMap",        numRows+1 );
//    OrdinalVector columnIndices( "columnIndices", nnz       );
//    ScalarVector  entries      ( "entries",       nnz       );
//    
//    Kokkos::parallel_for("initialize sample problem 3", numRows, LAMBDA_EXPRESSION(int row)
//    {
//      if (row == 0)
//      {
//        rowMap(0) = 0;
//        rowMap(1) = 1;
//        
//        columnIndices(0) = 0;
//        
//        entries(0) = 1.0;
//        
//        xExpected(0) = 0.0;
//        b(0) = 0.0;
//      }
//      else if (row == 1)
//      {
//        rowMap(1) = 1;
//        rowMap(2) = 2;
//        
//        columnIndices(1) = 1;
//        
//        entries(1) =  4.0;
//        
//        xExpected(1) = 1.0 / 48.0;
//        b(1) =  1.0 / 12.0;
//      }
//      else if (row == 2)
//      {
//        rowMap(2) = 2;
//        rowMap(3) = nnz;
//        
//        columnIndices(2) = 2;
//        entries(2) = 1.0;
//        
//        xExpected(2) = 1.0 / 6.0;
//        b(2) = 1.0 / 6.0;
//      }
//    });
//    
//    CrsMatrix A(rowMap, columnIndices, entries);
//    LinearProblem problem(A, xActual, b);
//    int result = problem.solve();
//    
//    TEST_EQUALITY(0, result);
//    
//    double tol = 1e-12;
//    testFloatingEquality(xExpected, xActual, tol, out, success);
//  }
} // namespace
