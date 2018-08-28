//
//  CrsMatrix.hpp
//
//
//  Created by Roberts, Nathan V on 6/20/17.
//
//

#ifndef LGR_CRS_MATRIX_HPP
#define LGR_CRS_MATRIX_HPP

#include "LGRLambda.hpp"
#include "LGR_Types.hpp"

#include <Kokkos_Core.hpp>

namespace lgr {
template <
    class Ordinal,
    class RowMapEntryType>
class CrsMatrix;

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
        RowMapEntryType>::ScalarVector b);

template <
    class Ordinal,
    class RowMapEntryType>
class CrsMatrix {
 public:
  typedef Kokkos::View<Ordinal*, MemSpace>         OrdinalVector;
  typedef Kokkos::View<Scalar*, MemSpace>          ScalarVector;
  typedef Kokkos::View<RowMapEntryType*, MemSpace> RowMapVector;

 private:
  RowMapVector  _rowMap;
  OrdinalVector _columnIndices;
  ScalarVector  _entries;

  int _blockSizeRow, _blockSizeCol;
  bool _isBlockMatrix;

 public:
  decltype(_isBlockMatrix) isBlockMatrix(){return _isBlockMatrix;}
  decltype(_blockSizeRow)  blockSizeRow(){return _blockSizeRow;}
  decltype(_blockSizeCol)  blockSizeCol(){return _blockSizeCol;}

  CrsMatrix() {}

  CrsMatrix(
      RowMapVector rowmap, OrdinalVector colIndices, ScalarVector entres,
      int blkSizeCol=1, int blkSizeRow=1)
      : _rowMap(rowmap), _columnIndices(colIndices), _entries(entres),
        _blockSizeRow(blkSizeRow), _blockSizeCol(blkSizeCol),
        _isBlockMatrix(_blockSizeRow*_blockSizeCol > 1) {}

  KOKKOS_INLINE_FUNCTION RowMapVector rowMap() { return _rowMap; }
  KOKKOS_INLINE_FUNCTION OrdinalVector columnIndices() {
    return _columnIndices;
  }
  KOKKOS_INLINE_FUNCTION ScalarVector entries() { return _entries; }

  // const versions:
  KOKKOS_INLINE_FUNCTION const RowMapVector rowMap() const { return _rowMap; }
  KOKKOS_INLINE_FUNCTION const OrdinalVector columnIndices() const {
    return _columnIndices;
  }
  KOKKOS_INLINE_FUNCTION const ScalarVector entries() const { return _entries; }

  void Apply(const ScalarVector x, const ScalarVector b) {
    ApplyCrsMatrix<Ordinal, RowMapEntryType>(
        *this, x, b);
  }
};

#define LGR_EXPL_INST_DECL(Ordinal, RowMapEntryType) \
extern template \
void ApplyCrsMatrix( \
    const CrsMatrix<Ordinal, RowMapEntryType> A, \
    const typename CrsMatrix< \
        Ordinal, \
        RowMapEntryType>::ScalarVector x, \
    const typename CrsMatrix< \
        Ordinal, \
        RowMapEntryType>::ScalarVector b);
LGR_EXPL_INST_DECL(int, int)
#undef LGR_EXPL_INST_DECL

}  // namespace lgr

#endif /* CrsMatrix_h */
