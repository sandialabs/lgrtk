#include "Basis.hpp"

namespace lgr {

Basis::Basis(int spaceDim) : _spaceDim(spaceDim) {}

int Basis::basisCardinality() { return _spaceDim + 1; }

void Basis::getRefCoords(PointsView coords) {
  /*
     Reference simplex: the first nodes lie on coordinate axes, in x/y/z order, at unit distance from the origin.
                        The final node is at the origin.  So, in 3D:
                        node 0: (1,0,0)
                        node 1: (0,1,0)
                        node 2: (0,0,1)
                        node 3: (0,0,0)
     It is worth noting that these are rearranged from Intrepid's ordering, which has the origin node first.
     The reason for this is that it allows us to accumuluate for the values at the origin as we iterate through
     the other nodes.  It's not clear this will ever make much difference in terms of computations, but it is
     aesthetically a little nicer.
     */
  Kokkos::deep_copy(coords, Scalar(0.0));  // initialize to 0
  Kokkos::parallel_for(_spaceDim, LAMBDA_EXPRESSION(int basisOrdinal) {
    coords(basisOrdinal, basisOrdinal) = 1.0;
  });
}

void Basis::getValues(PointsView refCoords, ValuesView values) {
  int numPoints = refCoords.extent(0);
  int spaceDim = refCoords.extent(1);
  Kokkos::parallel_for(numPoints, LAMBDA_EXPRESSION(int ptOrdinal) {
    // at point (x,y,z), field 0 = x
    //                   field 1 = y
    //                   field 2 = z
    //                   field 3 = 1 - x - y - z
    Scalar finalValue = 1.0;
    for (int basisOrdinal = 0; basisOrdinal < spaceDim; basisOrdinal++) {
      values(basisOrdinal, ptOrdinal) = refCoords(ptOrdinal, basisOrdinal);
      finalValue -= refCoords(ptOrdinal, basisOrdinal);
    }
    values(spaceDim, ptOrdinal) = finalValue;
  });
}

void Basis::getGradientValues(PointsView refCoords, GradientView gradientValues) {
  // derivative values are constant
  int numPoints = refCoords.extent(0);
  Kokkos::deep_copy(gradientValues, Scalar(0.0));  // initialize to 0
  Kokkos::parallel_for(numPoints, LAMBDA_EXPRESSION(int ptOrdinal) {
    // in 3D, gradient field 0 = ( 1 ,0, 0)
    //                 field 1 = ( 0, 1, 0)
    //                 field 2 = ( 0, 0, 1)
    //                 field 3 = (-1,-1,-1)
    for (int basisOrdinal = 0; basisOrdinal < _spaceDim; basisOrdinal++) {
      gradientValues(basisOrdinal, ptOrdinal, basisOrdinal) = 1.0;
    }
    for (int d = 0; d < _spaceDim; d++) {
      gradientValues(_spaceDim, ptOrdinal, d) = -1.0;
    }
  });
}

}  // namespace lgr
