#include "CrsMatrix.hpp"

namespace lgr {

// For CrsMatrix A, sets b := Ax
template <
    class Ordinal,
    class RowMapEntryType>
void ApplyCrsMatrix(
    const CrsMatrix<Ordinal, RowMapEntryType> A,
    const typename CrsMatrix<
        Ordinal,
        RowMapEntryType>::ScalarVector x,
    const typename CrsMatrix<
        Ordinal,
        RowMapEntryType>::ScalarVector b) {
  auto rowMap = A.rowMap();
  auto numRows = rowMap.size() - 1;
  auto columnIndices = A.columnIndices();
  auto entries = A.entries();
  Kokkos::parallel_for(
      Kokkos::RangePolicy<int>(0, numRows),
      LAMBDA_EXPRESSION(int rowOrdinal) {
        auto   rowStart = rowMap(rowOrdinal);
        auto   rowEnd = rowMap(rowOrdinal + 1);
        Scalar sum = 0.0;
        for (auto entryIndex = rowStart; entryIndex < rowEnd; entryIndex++) {
          auto columnIndex = columnIndices(entryIndex);
          sum += entries(entryIndex) * x(columnIndex);
        }
        b(rowOrdinal) = sum;
      },
      "CrsMatrix Apply()");
}

#define LGR_EXPL_INST(Ordinal, RowMapEntryType) \
template \
void ApplyCrsMatrix( \
    const CrsMatrix<Ordinal, RowMapEntryType> A, \
    const typename CrsMatrix< \
        Ordinal, \
        RowMapEntryType>::ScalarVector x, \
    const typename CrsMatrix< \
        Ordinal, \
        RowMapEntryType>::ScalarVector b);
LGR_EXPL_INST(int, int)
#undef LGR_EXPL_INST

}  // namespace lgr
