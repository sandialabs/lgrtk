#include "MatrixIO.hpp"

namespace lgr {

TempSetScientific::TempSetScientific(std::ostream& out)
    : out_(out), originalFlags_(out.flags()) {
  out << std::scientific;

  // this is the bit where we don't do things as nicely as Teuchos:
  // we hardcode a precision, whereas Teuchos has one that depends on the scalar type
  const int prec = 17;

  out.precision(prec);
}

TempSetScientific::~TempSetScientific() { out_.flags(originalFlags_); }

template <class Ordinal, class SizeType>
CrsMatrix<Ordinal, SizeType>
MatrixIO<Ordinal, SizeType>::
readSparseMatlabMatrix(std::istream& inStream) {
  /*
     Expects an input stream entirely consisting of lines of this form:
     row1 col1 value1
     row2 col2 value2
     ...
     */

  std::string line;

  // outer key: row #
  // inner key: col #
  // inner value: matrix entry
  std::map<Ordinal, std::map<Ordinal, Scalar>> rowsToColumnValues;

  int nnz = 0;

  while (std::getline(inStream, line)) {
    std::istringstream iss(line);
    Ordinal            row, col;
    Scalar             value;

    iss >> row;
    iss >> col;
    iss >> value;

    // subtract 1 from row and col because MATLAB is 1-based and we're 0-based:
    row--;
    col--;

    bool redundantEntry =
        rowsToColumnValues[row].find(col) != rowsToColumnValues[row].end();
    if (!redundantEntry) {
      nnz++;
    } else {
      std::cout << "Warning: redundant entries for row " << row << ", col "
                << col;
      std::cout << ".  (will use the last value specified)\n";
    }
    rowsToColumnValues[row][col] = value;
  }

  int rowCount = rowsToColumnValues.size();

  typedef Kokkos::View<Ordinal*, MemSpace>  OrdinalVector;
  typedef Kokkos::View<Scalar*, MemSpace>   ScalarVector;
  typedef Kokkos::View<SizeType*, MemSpace> SizeTypeVector;

  SizeTypeVector rowMap(
      "rowMap", rowCount + 1);  // entry offsets for each row
  OrdinalVector columnIndices(
      "columnIndices", nnz);             // column indices for each entry
  ScalarVector entries("entries", nnz);  // matrix values

  typename SizeTypeVector::HostMirror rowMapHost =
      Kokkos::create_mirror_view(rowMap);
  typename OrdinalVector::HostMirror columnIndicesHost =
      Kokkos::create_mirror_view(columnIndices);
  typename ScalarVector::HostMirror entriesHost =
      Kokkos::create_mirror_view(entries);

  int entryOffset = 0;
  rowMapHost(0) = 0;
  for (int row = 0; row < rowCount; row++) {
    int numColsForRow = rowsToColumnValues[row].size();
    rowMapHost(row + 1) = rowMapHost(row) + numColsForRow;
    for (auto entry : rowsToColumnValues[row]) {
      Ordinal columnIndex = entry.first;
      Scalar  value = entry.second;
      columnIndicesHost(entryOffset) = columnIndex;
      entriesHost(entryOffset) = value;
      entryOffset++;
    }
  }

  Kokkos::deep_copy(rowMap, rowMapHost);
  Kokkos::deep_copy(columnIndices, columnIndicesHost);
  Kokkos::deep_copy(entries, entriesHost);

  return CrsMatrix<Ordinal, SizeType>(rowMap, columnIndices, entries);
}

template <class Ordinal, class SizeType>
void
MatrixIO<Ordinal, SizeType>::
writeSparseMatlabMatrix(
    std::ostream&                                          out,
    CrsMatrix<Ordinal, SizeType> matrix) {
  // Make the output stream write floating-point numbers in
  // scientific notation.  It will politely put the output
  // stream back to its state on input, when this scope
  // terminates.
  //      Teuchos::MatrixMarket::details::SetScientific<Scalar> sci(out);
  TempSetScientific sci(out);

  auto rowMapHost = Kokkos::create_mirror_view(matrix.rowMap());
  auto columnIndicesHost = Kokkos::create_mirror_view(matrix.columnIndices());
  auto entriesHost = Kokkos::create_mirror_view(matrix.entries());

  Kokkos::deep_copy(rowMapHost, matrix.rowMap());
  Kokkos::deep_copy(columnIndicesHost, matrix.columnIndices());
  Kokkos::deep_copy(entriesHost, matrix.entries());

  SizeType rowCount = rowMapHost.size() - 1;
  int      entry = 0;
  for (SizeType row = 0; row < rowCount; row++) {
    int colsForRow = rowMapHost(row + 1) - rowMapHost(row);
    for (int colOrdinal = 0; colOrdinal < colsForRow; colOrdinal++) {
      int    col = columnIndicesHost(entry);
      Scalar value = entriesHost(entry);
      out << row + 1 << " " << col + 1 << " " << value << "\n";
      entry++;
    }
  }
}

template <class Ordinal, class SizeType>
void
MatrixIO<Ordinal, SizeType>::
writeDenseMatlabVector(
    std::ostream& out, Kokkos::View<Scalar*, Layout, MemSpace> vectorView) {
  // Make the output stream write floating-point numbers in
  // scientific notation.  It will politely put the output
  // stream back to its state on input, when this scope
  // terminates.
  //      Teuchos::MatrixMarket::details::SetScientific<Scalar> sci(out);
  TempSetScientific sci(out);

  typedef Kokkos::View<Scalar*, Layout, MemSpace> ScalarVector;
  typename ScalarVector::HostMirror               vectorHost =
      Kokkos::create_mirror_view(vectorView);

  Kokkos::deep_copy(vectorHost, vectorView);

  SizeType rowCount = vectorView.size();
  for (SizeType row = 0; row < rowCount; row++) {
    int    col = 0;
    Scalar value = vectorHost(row);
    out << row + 1 << " " << col + 1 << " " << value << "\n";
  }
}

template class MatrixIO<int, int>;

}  // namespace lgr
