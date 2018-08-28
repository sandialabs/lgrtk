#include "Teuchos_UnitTestHarness.hpp"

#include "LGRTestHelpers.hpp"

#include "CrsMatrix.hpp"
#include "MatrixIO.hpp"

#include <sstream>

using namespace lgr;

namespace {
  
  typedef int Ordinal;
  typedef int SizeType;
  
  typedef CrsMatrix<Ordinal, SizeType>          CrsMatrix;
  typedef MatrixIO <Ordinal, SizeType>          MatrixIO;
  
  typedef Kokkos::View<Ordinal*,  MemSpace> OrdinalVector;
  typedef Kokkos::View<Scalar* ,  MemSpace> ScalarVector;
  typedef Kokkos::View<SizeType*, MemSpace> SizeTypeVector;
  
  CrsMatrix sampleCrsMatrix(int numRows)
  {
    // let's just do a fixed 5 entries per row.
    // we'll include some big entries and some small entries
    using namespace std;
    vector<Scalar> values = {1e-13, 1e+13, 1.0, 2.15e-6, 3.14e+6}; // for each row
    vector<int> colOrdinalOffsets = {-2, -1, 0, 1, 2}; // we allow negative indices here
    int nnz = values.size() * numRows;
    
    SizeTypeVector rowMap      ( "rowMap",        numRows+1 );
    OrdinalVector columnIndices( "columnIndices", nnz       );
    ScalarVector  entries      ( "entries",       nnz       );
    
    SizeTypeVector::HostMirror rowMapHost       = Kokkos::create_mirror_view( rowMap );
    OrdinalVector::HostMirror columnIndicesHost = Kokkos::create_mirror_view( columnIndices );
    ScalarVector::HostMirror  entriesHost       = Kokkos::create_mirror_view( entries );
    
    int entryOffset = 0;
    for (int row=0; row<numRows; row++)
    {
      rowMapHost(row+1) = rowMapHost(row) + values.size();
      int colOrdinal = 0; // in this row
      for (Scalar value : values)
      {
        columnIndicesHost(entryOffset) = row + colOrdinalOffsets[colOrdinal];
        entriesHost      (entryOffset) = value;
        entryOffset++;
        colOrdinal++;
      }
    }
    
    Kokkos::deep_copy( rowMap,        rowMapHost);
    Kokkos::deep_copy( columnIndices, columnIndicesHost);
    Kokkos::deep_copy( entries,       entriesHost );
    
    return CrsMatrix(rowMap, columnIndices, entries);
  }
  
  TEUCHOS_UNIT_TEST( MatrixIO, CrsMatrixWriteAndReadMatlab )
  {
    // verify that for a sample CrsMatrix (with some small values) remains the same after a
    // write/read cycle
    int numRows = 8;
    CrsMatrix matrix = sampleCrsMatrix(numRows);
    
    using namespace std;
    
    ostringstream ss;
    MatrixIO::writeSparseMatlabMatrix(ss, matrix);
    
    string matrixString = ss.str();
    istringstream iss(matrixString);
    
//    out << "matrixString:\n" << matrixString << std::endl;
    
    CrsMatrix matrixOut = MatrixIO::readSparseMatlabMatrix(iss);
    
    double tol = 1e-15;
    testFloatingEquality<Ordinal, SizeType>(matrix,matrixOut,tol,out,success);
  }
} // namespace
