#ifndef STATIC_TYPES_HPP
#define STATIC_TYPES_HPP

#include <map>

#include "CrsMatrix.hpp"
#include "LGR_Types.hpp"

namespace lgr {

  using OrdinalType       = int;
  using RowMapEntryType   = int;

  using CrsMatrixType      = typename lgr::CrsMatrix<OrdinalType, RowMapEntryType>;
  using LocalOrdinalVector = typename Kokkos::View<OrdinalType*, MemSpace>;

  template <typename ScalarType> 
  using ScalarVectorT = typename Kokkos::View<ScalarType*, MemSpace>;
  using ScalarVector  = ScalarVectorT<Scalar>;

  template <typename ScalarType> 
  using ScalarMultiVectorT = typename Kokkos::View<ScalarType**, Kokkos::LayoutRight, MemSpace>;
  using ScalarMultiVector  = ScalarMultiVectorT<Scalar>;

  template <typename ScalarType> 
  using ScalarArray3DT = typename Kokkos::View<ScalarType***, Kokkos::LayoutRight, MemSpace>;
  using ScalarArray3D  = ScalarArray3DT<Scalar>;

} // namespace lgr

#endif
