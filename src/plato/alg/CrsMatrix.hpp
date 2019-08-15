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
  int  mNumRowsPerBlock;
  int  mNumColsPerBlock;
  bool mIsBlockMatrix;

 public:
  decltype(mIsBlockMatrix)  isBlockMatrix()     const { return mIsBlockMatrix; }
  decltype(mNumRowsPerBlock) numRowsPerBlock()  const { return mNumRowsPerBlock; }
  decltype(mNumColsPerBlock) numColsPerBlock()  const { return mNumColsPerBlock; }

  decltype(mNumRows) numRows() const
  { return (mNumRows != -1) ? mNumRows : throw std::logic_error("requested unset value"); }

  decltype(mNumCols) numCols() const
  { return (mNumCols != -1) ? mNumCols : throw std::logic_error("requested unset value"); }

  CrsMatrix() :
            mNumRows         (-1),
            mNumCols         (-1),
            mNumRowsPerBlock ( 1),
            mNumColsPerBlock ( 1),
            mIsBlockMatrix (false) {}

  CrsMatrix( int           aNumRows,
             int           aNumCols,
             int           aNumRowsPerBlock,
             int           aNumColsPerBlock
           ) :
            mNumRows         (aNumRows),
            mNumCols         (aNumCols),
            mNumRowsPerBlock (aNumRowsPerBlock),
            mNumColsPerBlock (aNumColsPerBlock),
            mIsBlockMatrix   (mNumColsPerBlock*mNumRowsPerBlock > 1) {}

  CrsMatrix( RowMapVector  aRowmap,
             OrdinalVector aColIndices,
             ScalarVector  aEntries,
             int           aNumRowsPerBlock=1,
             int           aNumColsPerBlock=1
           ) :
            mRowMap          (aRowmap),
            mColumnIndices   (aColIndices),
            mEntries         (aEntries),
            mNumRows         (-1),
            mNumCols         (-1),
            mNumRowsPerBlock (aNumRowsPerBlock),
            mNumColsPerBlock (aNumColsPerBlock),
            mIsBlockMatrix   (mNumColsPerBlock*mNumRowsPerBlock > 1) {}

  CrsMatrix( RowMapVector  aRowmap,
             OrdinalVector aColIndices,
             ScalarVector  aEntries,
             int           aNumRows,
             int           aNumCols,
             int           aNumRowsPerBlock,
             int           aNumColsPerBlock
          ) :
            mRowMap          (aRowmap),
            mColumnIndices   (aColIndices),
            mEntries         (aEntries),
            mNumRows         (aNumRows),
            mNumCols         (aNumCols),
            mNumRowsPerBlock (aNumRowsPerBlock),
            mNumColsPerBlock (aNumColsPerBlock),
            mIsBlockMatrix   (mNumColsPerBlock*mNumRowsPerBlock > 1) {}

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
