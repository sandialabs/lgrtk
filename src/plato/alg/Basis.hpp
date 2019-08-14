//
//  Basis.hpp
//
//
//  Created by Roberts, Nathan V on 6/20/17.
//
//

#ifndef PLATO_BASIS_HPP
#define PLATO_BASIS_HPP

#include <Kokkos_Core.hpp>

#include "plato/alg/PlatoLambda.hpp"
#include "plato/PlatoTypes.hpp"

namespace Plato {

class Basis {
 private:
  using Layout = Kokkos::LayoutRight;
  using CoordScalar = Scalar;
  typedef Kokkos::View<Scalar**, Layout, MemSpace>  ValuesView;      // (F,P)
  typedef Kokkos::View<Scalar***, Layout, MemSpace> GradientView;    // (F,P,D)
  typedef Kokkos::View<CoordScalar**, Layout, MemSpace> PointsView;  // (P,D)

  int _spaceDim;

 public:
  Basis(int spaceDim);

  int basisCardinality();

  void getRefCoords(PointsView coords);

  void getValues(PointsView refCoords, ValuesView values);

  void getGradientValues(PointsView refCoords, GradientView gradientValues);

};

}  // namespace Plato

#endif /* Basis_h */
