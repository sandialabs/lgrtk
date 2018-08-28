#include "Teuchos_UnitTestHarness.hpp"

#include "LGRTestHelpers.hpp"

#include "CrsMatrix.hpp"
#include "MatrixIO.hpp"

#include <sstream>

using namespace lgr;
using namespace std;

namespace {
  
  typedef double                                                                       Scalar;
  typedef int                                                                          Ordinal;
  typedef Kokkos::DefaultExecutionSpace                                                MemSpace;
  typedef int    SizeType;
  
  typedef CrsMatrix<Ordinal, SizeType>          CrsMatrix;
  typedef MatrixIO <Ordinal, SizeType>          MatrixIO;
  
  typedef Kokkos::View<Ordinal*,  MemSpace> OrdinalVector;
  typedef Kokkos::View<Scalar* ,  MemSpace> ScalarVector;
  typedef Kokkos::View<SizeType*, MemSpace> SizeTypeVector;
  
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
  
  TEUCHOS_UNIT_TEST( CrsMatrix, Multiply )
  {
    // verify that for a sample CrsMatrix (with some small values) remains the same after a
    // write/read cycle
    vector<int> rowCounts = {1,2,3,4,5,6};

    for (int numRows : rowCounts)
    {
      CrsMatrix A = sampleCrsMatrix(numRows); // [[2 1]; [1 2]]
      ScalarVector x = sampleLHS(numRows);
      ScalarVector bExpected = sampleRHS(numRows);
      
      ScalarVector b = ScalarVector("b",numRows);
      
      A.Apply(x,b);
      
      double tol = 1e-15;
      testFloatingEquality<Scalar, ScalarVector>(bExpected,b,tol,out,success);
    }
  }
} // namespace
