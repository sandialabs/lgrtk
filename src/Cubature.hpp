//
//  Cubature_Simplex.hpp
//
//
//  Created by Roberts, Nathan V on 6/21/17.
//
//

#ifndef LGR_CUBATURE_HPP
#define LGR_CUBATURE_HPP

#include "LGR_Types.hpp"
#include <Kokkos_Core.hpp>

/*
 Define quadrature (cubature, in Intrepid parlance) rules for the simplex.
 */

namespace lgr {

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

}  // namespace lgr

#endif /* Cubature_Simplex_h */
