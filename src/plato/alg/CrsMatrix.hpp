//
//  CrsMatrix.hpp
//
//
//  Created by Roberts, Nathan V on 6/20/17.
//
//

#ifndef PLATO_CRS_MATRIX_HPP
#define PLATO_CRS_MATRIX_HPP

#include "plato/alg/PlatoLambda.hpp"
#include "plato/PlatoTypes.hpp"

#include <Kokkos_Core.hpp>

namespace Plato {

template < class Ordinal = Plato::OrdinalType >
class CrsMatrix {
 public:
  typedef Kokkos::View<Ordinal*, MemSpace> OrdinalVector;
  typedef Kokkos::View<Scalar*,  MemSpace> ScalarVector;
  typedef Kokkos::View<Ordinal*, MemSpace> RowMapVector;

 private:

  RowMapVector  mRowMap;
  OrdinalVector mColumnIndices;
  ScalarVector  mEntries;

  int  mNumRows;
  int  mNumCols;
  int  mBlockSizeRow;
  int  mBlockSizeCol;
  bool mIsBlockMatrix;

 public:
  decltype(mIsBlockMatrix) isBlockMatrix() const { return mIsBlockMatrix; }
  decltype(mBlockSizeRow)  numRowsPerBlock()  const { return mBlockSizeRow; }
  decltype(mBlockSizeCol)  numColsPerBlock()  const { return mBlockSizeCol; }

  decltype(mNumRows) numRows() const
  { return (mNumRows != -1) ? mNumRows : throw std::logic_error("requested unset value"); }

  decltype(mNumCols) numCols() const
  { return (mNumCols != -1) ? mNumCols : throw std::logic_error("requested unset value"); }

  CrsMatrix() :
            mNumRows       (-1),
            mNumCols       (-1),
            mBlockSizeRow  ( 1),
            mBlockSizeCol  ( 1),
            mIsBlockMatrix (false) {}

  CrsMatrix( int           aNumRows,
             int           aNumCols,
             int           aBlkSizeRow,
             int           aBlkSizeCol
           ) :
            mNumRows       (aNumRows),
            mNumCols       (aNumCols),
            mBlockSizeRow  (aBlkSizeRow),
            mBlockSizeCol  (aBlkSizeCol),
            mIsBlockMatrix (mBlockSizeRow*mBlockSizeCol > 1) {}

  CrsMatrix( RowMapVector  aRowmap,
             OrdinalVector aColIndices,
             ScalarVector  aEntries,
             int           aBlkSizeRow=1,
             int           aBlkSizeCol=1
           ) :
            mRowMap        (aRowmap),
            mColumnIndices (aColIndices),
            mEntries       (aEntries),
            mNumRows       (-1),
            mNumCols       (-1),
            mBlockSizeRow  (aBlkSizeRow),
            mBlockSizeCol  (aBlkSizeCol),
            mIsBlockMatrix (mBlockSizeRow*mBlockSizeCol > 1) {}

  CrsMatrix( RowMapVector  aRowmap,
             OrdinalVector aColIndices,
             ScalarVector  aEntries,
             int           aNumRows,
             int           aNumCols,
             int           aBlkSizeRow,
             int           aBlkSizeCol
          ) :
            mRowMap        (aRowmap),
            mColumnIndices (aColIndices),
            mEntries       (aEntries),
            mNumRows       (aNumRows),
            mNumCols       (aNumCols),
            mBlockSizeRow  (aBlkSizeRow),
            mBlockSizeCol  (aBlkSizeCol),
            mIsBlockMatrix (mBlockSizeRow*mBlockSizeCol > 1) {}

  KOKKOS_INLINE_FUNCTION decltype(mRowMap)        rowMap()        { return mRowMap; }
  KOKKOS_INLINE_FUNCTION decltype(mColumnIndices) columnIndices() { return mColumnIndices; }
  KOKKOS_INLINE_FUNCTION decltype(mEntries)       entries()       { return mEntries; }

  KOKKOS_INLINE_FUNCTION const decltype(mRowMap)        rowMap()        const { return mRowMap; }
  KOKKOS_INLINE_FUNCTION const decltype(mColumnIndices) columnIndices() const { return mColumnIndices; }
  KOKKOS_INLINE_FUNCTION const decltype(mEntries)       entries()       const { return mEntries; }

  KOKKOS_INLINE_FUNCTION void setRowMap       (decltype(mRowMap)        aRowMap)        { mRowMap = aRowMap; }
  KOKKOS_INLINE_FUNCTION void setColumnIndices(decltype(mColumnIndices) aColumnIndices) { mColumnIndices = aColumnIndices; }
  KOKKOS_INLINE_FUNCTION void setEntries      (decltype(mEntries)       aEntries)       { mEntries = aEntries; }

};

}  // namespace Plato

#endif /* CrsMatrix_h */
