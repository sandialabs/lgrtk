/*
 * PlatoStaticsTypes.hpp
 *
 *  Created on: Jul 12, 2018
 */

#ifndef SRC_PLATO_PLATOSTATICSTYPES_HPP_
#define SRC_PLATO_PLATOSTATICSTYPES_HPP_

#include <map>

#include "CrsMatrix.hpp"
#include "PlatoTypes.hpp"

namespace Plato
{

using OrdinalType     = int;
using RowMapEntryType = int;

using CrsMatrixType      = typename lgr::CrsMatrix<Plato::OrdinalType, Plato::RowMapEntryType>;
using LocalOrdinalVector = typename Kokkos::View<Plato::OrdinalType*, Plato::MemSpace>;

template <typename ScalarType>
using ScalarVectorT = typename Kokkos::View<ScalarType*,Plato:: MemSpace>;
using ScalarVector  = ScalarVectorT<Plato::Scalar>;

template <typename ScalarType>
using ScalarMultiVectorT = typename Kokkos::View<ScalarType**, Kokkos::LayoutRight, Plato::MemSpace>;
using ScalarMultiVector  = ScalarMultiVectorT<Plato::Scalar>;

template <typename ScalarType>
using ScalarArray3DT = typename Kokkos::View<ScalarType***, Kokkos::LayoutRight, Plato::MemSpace>;
using ScalarArray3D  = ScalarArray3DT<Plato::Scalar>;

struct DataMap
{
  std::map<std::string, Plato::ScalarVector> scalarVectors;
  std::map<std::string, Plato::ScalarMultiVector> scalarMultiVectors;
  std::map<std::string, Plato::ScalarArray3D> scalarArray3Ds;
};
// struct DataMap

} // namespace Plato

#endif /* SRC_PLATO_PLATOSTATICSTYPES_HPP_ */
