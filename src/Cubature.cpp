//
//  Cubature.cpp
//
//
//  Created by Roberts, Nathan V on 6/21/17.
//
//

#include "Cubature.hpp"

#include <Teuchos_TestForException.hpp>

namespace lgr {

int Cubature::getNumCubaturePoints(
    int spaceDim, int degree) {
  switch (spaceDim) {
    case 1:
      if (degree <= 1) {
        // one-point rule at the centroid
        return 1;
      } else if (degree <= 3) {
        // two-point rule
        return 2;
      } else {
        // TODO: add support for higher-order integration
        TEUCHOS_TEST_FOR_EXCEPTION(
            true, std::invalid_argument, "Unsupported degree");
      }
    case 2:
      if (degree <= 1) {
        // one-point rule at the centroid
        return 1;
      } else if (degree == 2) {
        // three-point rule
        return 3;
      } else if (degree == 3) {
        // four-point rule
        return 4;
      } else {
        // TODO: add support for higher-order integration
        TEUCHOS_TEST_FOR_EXCEPTION(
            true, std::invalid_argument, "Unsupported degree");
      }
    case 3:
      if (degree <= 1) {
        // one-point rule at the centroid
        return 1;
      } else if (degree == 2) {
        // four-point rule
        return 4;
      } else if (degree == 3) {
        // five-point rule
        return 5;
      } else {
        // TODO: add support for higher-order integration
        TEUCHOS_TEST_FOR_EXCEPTION(
            true, std::invalid_argument, "Unsupported degree");
      }
    default:
      TEUCHOS_TEST_FOR_EXCEPTION(
          true, std::invalid_argument, "Unsupported spaceDim");
  }
}

void Cubature::getCubature(
    int                 spaceDim,
    int                 degree,
    const RefPointsView points,
    const WeightsView   weights) {
  int numCubaturePoints = getNumCubaturePoints(spaceDim, degree);

  TEUCHOS_TEST_FOR_EXCEPTION(
      points.extent(0) != unsigned(numCubaturePoints), std::invalid_argument,
      "points should have dimensions(numCubPoints, spaceDim)");
  TEUCHOS_TEST_FOR_EXCEPTION(
      points.extent(1) != unsigned(spaceDim), std::invalid_argument,
      "points should have dimensions(numCubPoints, spaceDim)");
  TEUCHOS_TEST_FOR_EXCEPTION(
      weights.extent(0) != unsigned(numCubaturePoints),
      std::invalid_argument, "weights should have dimensions(numCubPoints)");
  typename RefPointsView::HostMirror pointsHost =
      Kokkos::create_mirror_view(points);
  typename WeightsView::HostMirror weightsHost =
      Kokkos::create_mirror_view(weights);

  switch (spaceDim) {
    case 1:
      /*
         For spaceDim == 1, we shift things from the [-1,1] interval that Intrepid defines as its reference cell
         to a [0,1] interval that is consistent with our simplex treatment in higher dimensions.
         
         That means coordinates are transformed by x -> (x + 1.0) / 2.0, and weightsHost are cut in half,
         compared with the pointsHost and weightsHost defined in Intrepid_CubatureDirectLineGaussDef.hpp
         */
      if (degree <= 1) {
        // one-point rule at the centroid
        pointsHost(0, 0) = 0.5;
        weightsHost(0) = 1.0;  // "volume" of the simplex
      } else if (degree <= 3) {
        // two-point rule
        const Scalar sqrt3_3 = 5.773502691896257645091487805019574556476e-1;
        pointsHost(0, 0) = (sqrt3_3 + 1.0) / 2.0;
        pointsHost(1, 0) = (-sqrt3_3 + 1.0) / 2.0;

        weightsHost(0) = 0.5;
        weightsHost(1) = 0.5;
      } else {
        // TODO: add support for higher-order integration
        TEUCHOS_TEST_FOR_EXCEPTION(
            true, std::invalid_argument, "Unsupported degree");
      }
      break;
    case 2:
      if (degree <= 1) {
        // one-point rule at the centroid
        pointsHost(0, 0) = 1. / 3.;
        pointsHost(0, 1) = 1. / 3.;

        weightsHost(0) = 1. / 2.;  // "volume" of the simplex
      } else if (degree == 2) {
        // three-point rule
        pointsHost(0, 0) = 1. / 6.;
        pointsHost(0, 1) = 1. / 6.;

        pointsHost(1, 0) = 1. / 6.;
        pointsHost(1, 1) = 2. / 3.;

        pointsHost(2, 0) = 2. / 3.;
        pointsHost(2, 1) = 1. / 6.;

        weightsHost(0) = 1. / 6.;
        weightsHost(1) = 1. / 6.;
        weightsHost(2) = 1. / 6.;
      } else if (degree == 3) {
        // four-point rule
        pointsHost(0, 0) = 1. / 3.;
        pointsHost(0, 1) = 1. / 3.;

        pointsHost(1, 0) = 1. / 5.;
        pointsHost(1, 1) = 1. / 5.;

        pointsHost(2, 0) = 1. / 5.;
        pointsHost(2, 1) = 3. / 5.;

        pointsHost(3, 0) = 3. / 5.;
        pointsHost(3, 1) = 1. / 5.;

        weightsHost(0) = -9. / 32.;
        weightsHost(1) = 25. / 96.;
        weightsHost(2) = 25. / 96.;
        weightsHost(3) = 25. / 96.;
      } else {
        // TODO: add support for higher-order integration
        TEUCHOS_TEST_FOR_EXCEPTION(
            true, std::invalid_argument, "Unsupported degree");
      }
      break;
    case 3:
      if (degree <= 1) {
        // one-point rule at the centroid
        pointsHost(0, 0) = 1. / 4.;
        pointsHost(0, 1) = 1. / 4.;
        pointsHost(0, 2) = 1. / 4.;

        weightsHost(0) = 1. / 6.;  // "volume" of the simplex
      } else if (degree == 2) {
        // four-point rule
        pointsHost(0, 0) = 0.1381966011250105151795413165634361882280;
        pointsHost(0, 1) = 0.1381966011250105151795413165634361882280;
        pointsHost(0, 2) = 0.1381966011250105151795413165634361882280;

        pointsHost(1, 0) = 0.5854101966249684544613760503096914353161;
        pointsHost(1, 1) = 0.1381966011250105151795413165634361882280;
        pointsHost(1, 2) = 0.1381966011250105151795413165634361882280;

        pointsHost(2, 0) = 0.1381966011250105151795413165634361882280;
        pointsHost(2, 1) = 0.5854101966249684544613760503096914353161;
        pointsHost(2, 2) = 0.1381966011250105151795413165634361882280;

        pointsHost(3, 0) = 0.1381966011250105151795413165634361882280;
        pointsHost(3, 1) = 0.1381966011250105151795413165634361882280;
        pointsHost(3, 2) = 0.5854101966249684544613760503096914353161;

        weightsHost(0) = 1. / 24.;
        weightsHost(1) = 1. / 24.;
        weightsHost(2) = 1. / 24.;
        weightsHost(3) = 1. / 24.;
      } else if (degree == 3) {
        // five-point rule
        pointsHost(0, 0) = 1. / 4.;
        pointsHost(0, 1) = 1. / 4.;
        pointsHost(0, 2) = 1. / 4.;

        pointsHost(1, 0) = 1. / 6.;
        pointsHost(1, 1) = 1. / 6.;
        pointsHost(1, 2) = 1. / 6.;

        pointsHost(2, 0) = 1. / 6.;
        pointsHost(2, 1) = 1. / 6.;
        pointsHost(2, 2) = 1. / 2.;

        pointsHost(3, 0) = 1. / 6.;
        pointsHost(3, 1) = 1. / 2.;
        pointsHost(3, 2) = 1. / 6.;

        pointsHost(4, 0) = 1. / 2.;
        pointsHost(4, 1) = 1. / 6.;
        pointsHost(4, 2) = 1. / 6.;

        weightsHost(0) = -2. / 15.;
        weightsHost(1) = 3. / 40.;
        weightsHost(2) = 3. / 40.;
        weightsHost(3) = 3. / 40.;
        weightsHost(4) = 3. / 40.;
      } else {
        // TODO: add support for higher-order integration
        TEUCHOS_TEST_FOR_EXCEPTION(
            true, std::invalid_argument, "Unsupported degree");
      }
      break;
    default:
      TEUCHOS_TEST_FOR_EXCEPTION(
          true, std::invalid_argument, "Unsupported spaceDim");
  }
  // copy from host to device
  Kokkos::deep_copy(weights, weightsHost);
  Kokkos::deep_copy(points, pointsHost);
}

}  // namespace lgr
