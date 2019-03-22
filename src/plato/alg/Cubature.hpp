//
//  Cubature_Simplex.hpp
//
//
//  Created by Roberts, Nathan V on 6/21/17.
//
//

#ifndef PLATO_CUBATURE_HPP
#define PLATO_CUBATURE_HPP

#include "Plato_Types.hpp"
#include <Kokkos_Core.hpp>

/*
 Define quadrature (cubature, in Intrepid parlance) rules for the simplex.
 */

namespace Plato {

struct Cubature {
  typedef Kokkos::View<Scalar**, Kokkos::LayoutRight, MemSpace> RefPointsView;  // (P,D)
  typedef Kokkos::View<Scalar*, MemSpace>  WeightsView;    // (P)
  static void getCubature(
      int                 spaceDim,
      int                 degree,
      const RefPointsView points,
      const WeightsView   weights);
  static int getNumCubaturePoints(int spaceDim, int degree);
};

}  // namespace Plato

#endif /* Cubature_Simplex_h */
