//
//  MatrixIO.hpp
//
//
//  Created by Roberts, Nathan V on 6/29/17.
//
//

#ifndef LGR_MATRIX_IO_HPP
#define LGR_MATRIX_IO_HPP

#include <iostream>
#include <map>
// TODO: ask whether we can depend on this in lgr; right now we don't have access to it in white's Trilinos build.
//#include "Teuchos_MatrixMarket_SetScientific.hpp"

#include "CrsMatrix.hpp"
#include "Kokkos_Core.hpp"

namespace lgr {
/*
   The MatrixIO utilities are primarily intended for debugging.  (We are not too concerned with performance here; we are also not too concerned with robust error handling when e.g. the input stream is not as we expect it to be.)
   
   Right now, there is not support for MPI-distributed matrices, and all work is done in a single thread.
   
   */

/*
   The below class is just a placeholder, until we can depend on Teuchos::MatrixMarket::details::SetScientific.
   
   (If we for some reason can't, then we'd do well to imitate its full functionality.)
   */
class TempSetScientific {
 private:
  //! The output stream to which to apply flags.
  std::ostream& out_;

  //! The output stream's original flags.
  std::ios_base::fmtflags originalFlags_;

 public:
  TempSetScientific(std::ostream& out);
  ~TempSetScientific();
};

template <
    class Ordinal,
    class SizeType>
class MatrixIO {
 public:
  static CrsMatrix<Ordinal, SizeType>
  readSparseMatlabMatrix(std::istream& inStream);

  static void writeSparseMatlabMatrix(
      std::ostream&                                          out,
      CrsMatrix<Ordinal, SizeType> matrix);

  static void writeDenseMatlabVector(
      std::ostream& out, Kokkos::View<Scalar*, Layout, MemSpace> vectorView);
};

extern template class MatrixIO<int, int>;

}  // namespace lgr

#endif /* MatrixIO_h */
