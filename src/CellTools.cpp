//
//  CellTools_Simplex.hpp
//
//
//  Created by Roberts, Nathan V on 6/20/17.
//
//

#include "CellTools.hpp"

#include "LGRLambda.hpp"
#include "Basis.hpp"
#include "MatrixIO.hpp"  // for debugging output

#include <Omega_h_matrix.hpp>

#include "Teuchos_TestForException.hpp"

namespace lgr {

void CellTools::setJacobian(const JacobianView    jacobian,
    const RefPointsView   points,
    const PhysPointsView  cellWorkset,
    const RefGradientView workspace) {
  // The Jacobian we compute here matches the one that Intrepid computes.  This is the transpose
  // of the Jacobian as usually defined.  See
  //      https://trilinos.org/docs/dev/packages/intrepid/doc/html/cell_tools_page.html
  // for details on Intrepid's approach, which we imitate here.

  // A point worth noting: the gradients of the lowest-order simplex basis are constants,
  // and therefore independent of the point selected.  This means that the entire Jacobian
  // is likewise independent of the point selected, so that we should be able to speed up
  // computations by taking advantage of this fact.

  // For now, we proceed with the generic approach implemented in Intrepid (which works for
  // arbitrary cell topologies).  But we should keep this optimization in mind for the future;
  // we could have a "fused" computation that does FEM assembly in a way that takes advantage
  // of this.

  // (We can actually go a little further with the fusing idea: the gradients are zero in many components;
  //  we can avoid computing with or storing those zeros.)

  int numCells = jacobian.extent(0);
  int numPoints = jacobian.extent(1);
  // should match jacobian.dimension(3), points.extent(1), cellWorkset.extent(2)
  int spaceDim = jacobian.extent(2);

  // compute gradients:
  Basis basis(spaceDim);
  basis.getGradientValues(
      points,
      workspace);  // will be constant across its second (ptOrdinal) dimension
  // should match spaceDim + 1 for the simplex
  int numNodes = cellWorkset.extent(1);

  Kokkos::deep_copy(jacobian, Scalar(0.0));  // initialize to 0
  Kokkos::parallel_for(numCells, LAMBDA_EXPRESSION(int cellOrdinal) {
    // this is the loop we could eliminate for simplices in particular
    for (int ptOrdinal = 0; ptOrdinal < numPoints; ptOrdinal++) {
      for (int d1 = 0; d1 < spaceDim; d1++) {
        for (int d2 = 0; d2 < spaceDim; d2++) {
          for (int nodeOrdinal = 0; nodeOrdinal < numNodes; nodeOrdinal++) {
            jacobian(cellOrdinal, ptOrdinal, d1, d2) +=
                cellWorkset(cellOrdinal, nodeOrdinal, d1) *
                workspace(nodeOrdinal, ptOrdinal, d2);
          }
        }
      }
    }
  });
}

// this is our first effort at a point-independent Jacobian computation
template <int spaceDim>
void CellTools::setFusedJacobian(
    const FusedJacobianView<spaceDim> jacobian,
    const PhysPointsView              cellWorkset) {
  // The Jacobian we compute here matches the one that Intrepid computes, except that it omits the point dimension
  // from the container.  This is the transpose of the Jacobian as usually defined.  See
  //      https://trilinos.org/docs/dev/packages/intrepid/doc/html/cell_tools_page.html
  // for details on Intrepid's approach, which we imitate here.

  // A point worth noting: the gradients of the lowest-order simplex basis are constants,
  // and therefore independent of the point selected.  This means that the entire Jacobian
  // is likewise independent of the point selected, so that we should be able to speed up
  // computations by taking advantage of this fact.

  // (We can actually go a little further with the fusing idea: the gradients are zero in many components;
  //  we can avoid computing with or storing those zeros.)

  int numCells = jacobian.extent(0);

  Omega_h::Matrix<spaceDim, spaceDim> zeroMatrix;
  for (int d1 = 0; d1 < spaceDim; d1++)
    for (int d2 = 0; d2 < spaceDim; d2++) zeroMatrix[d1][d2] = 0.0;

  Kokkos::deep_copy(jacobian, zeroMatrix);  // initialize to 0
  Kokkos::parallel_for(numCells, LAMBDA_EXPRESSION(int cellOrdinal) {
    for (int d1 = 0; d1 < spaceDim; d1++) {
      //          // Each of the first _spaceDim basis fields has derivative 1 in the (basisOrdinal) component, 0 in the others.
      //          // Therefore, for each nodeOrdinal < _spaceDim, all but the d2 = (nodeOrdinal-1) contribution will vanish
      //          for (int nodeOrdinal=0; nodeOrdinal<_spaceDim; nodeOrdinal++)
      //          {
      //            jacobian(cellOrdinal,d1,nodeOrdinal) += 1.0 * cellWorkset(cellOrdinal,nodeOrdinal,d1);
      //          }
      //          // final basis field has derivative -1 in each dimension
      //          for (int d2=0; d2<spaceDim; d2++)
      //          {
      //            jacobian(cellOrdinal,d1,d2) += -1.0 * cellWorkset(cellOrdinal,spaceDim,d1); // final node contribution
      //          }

      // The above code may be easier to read, but I believe it is equivalent to this:
      for (int d2 = 0; d2 < spaceDim; d2++) {
        jacobian(cellOrdinal)[d1][d2] +=
            cellWorkset(cellOrdinal, d2, d1) -
            cellWorkset(cellOrdinal, spaceDim, d1);
      }
      // What's more, if that's correct, we should be able to change the += to =, and eliminate the deep_copy() above:
      // we only visit each entry once, so there is no need to accumulate.
    }
  });
}

void CellTools::setJacobianDet(
    const JacobianDetView jacobianDet, const JacobianView jacobian) {
  // jacobianDet: (C,P)
  // jacobian:    (C,P,D,D)
  auto numCells = jacobian.extent(0);
  int numPoints = int(jacobian.extent(1));
  int spaceDim = int(jacobian.extent(2));
  if (spaceDim == 1) {
    Kokkos::parallel_for(numCells, LAMBDA_EXPRESSION(int cellOrdinal) {
      for (int ptOrdinal = 0; ptOrdinal < numPoints; ptOrdinal++) {
        jacobianDet(cellOrdinal, ptOrdinal) =
            jacobian(cellOrdinal, ptOrdinal, 0, 0);
      }
    });
  } else if (spaceDim == 2) {
    Kokkos::parallel_for(numCells, LAMBDA_EXPRESSION(int cellOrdinal) {
      for (int ptOrdinal = 0; ptOrdinal < numPoints; ptOrdinal++) {
        jacobianDet(cellOrdinal, ptOrdinal) =
            jacobian(cellOrdinal, ptOrdinal, 0, 0) *
                jacobian(cellOrdinal, ptOrdinal, 1, 1) -
            jacobian(cellOrdinal, ptOrdinal, 0, 1) *
                jacobian(cellOrdinal, ptOrdinal, 1, 0);
      }
    });
  } else if (spaceDim == 3) {
    Kokkos::parallel_for(numCells, LAMBDA_EXPRESSION(int cellOrdinal) {
      for (int ptOrdinal = 0; ptOrdinal < numPoints; ptOrdinal++) {
        jacobianDet(cellOrdinal, ptOrdinal) =
            jacobian(cellOrdinal, ptOrdinal, 0, 0) *
                jacobian(cellOrdinal, ptOrdinal, 1, 1) *
                jacobian(cellOrdinal, ptOrdinal, 2, 2) +
            jacobian(cellOrdinal, ptOrdinal, 0, 1) *
                jacobian(cellOrdinal, ptOrdinal, 1, 2) *
                jacobian(cellOrdinal, ptOrdinal, 2, 0) +
            jacobian(cellOrdinal, ptOrdinal, 0, 2) *
                jacobian(cellOrdinal, ptOrdinal, 1, 0) *
                jacobian(cellOrdinal, ptOrdinal, 2, 1) -
            jacobian(cellOrdinal, ptOrdinal, 0, 0) *
                jacobian(cellOrdinal, ptOrdinal, 1, 2) *
                jacobian(cellOrdinal, ptOrdinal, 2, 1) -
            jacobian(cellOrdinal, ptOrdinal, 0, 1) *
                jacobian(cellOrdinal, ptOrdinal, 1, 0) *
                jacobian(cellOrdinal, ptOrdinal, 2, 2) -
            jacobian(cellOrdinal, ptOrdinal, 0, 2) *
                jacobian(cellOrdinal, ptOrdinal, 1, 1) *
                jacobian(cellOrdinal, ptOrdinal, 2, 0);
      }
    });
  }
}

template <int spaceDim>
void CellTools::setFusedJacobianDet(
    const FusedJacobianDetView        jacobianDet,
    const FusedJacobianView<spaceDim> jacobian) {
  int numCells = jacobian.extent(0);

  Kokkos::parallel_for(numCells, LAMBDA_EXPRESSION(int cellOrdinal) {
    jacobianDet(cellOrdinal) = Omega_h::determinant(jacobian(cellOrdinal));
  });
}

// TODO: write a unit test against this
template <int spaceDim>
void CellTools::setFusedJacobianInv(
    const FusedJacobianView<spaceDim> jacobianInv,
    const FusedJacobianView<spaceDim> jacobian) {
  int numCells = jacobian.extent(0);
  Kokkos::parallel_for(numCells, LAMBDA_EXPRESSION(int cellOrdinal) {
    jacobianInv(cellOrdinal) = Omega_h::invert(jacobian(cellOrdinal));
  });
}

// TODO: write a unit test against this
template <int spaceDim>
void CellTools::getPhysicalGradients(
    const PhysCellGradientView        cellGradients,
    const FusedJacobianView<spaceDim> jacobianInv) {
  // ref gradients in 3D are:
  //    field 0 = ( 1 ,0, 0)
  //    field 1 = ( 0, 1, 0)
  //    field 2 = ( 0, 0, 1)
  //    field 3 = (-1,-1,-1)

  // Therefore, when we multiply by the transpose jacobian inverse (which is what we do to compute physical gradients),
  // we have the following values:
  //    field 0 = row 0 of jacobianInv
  //    field 1 = row 1 of jacobianInv
  //    field 2 = row 2 of jacobianInv
  //    field 3 = negative sum of the three rows

  Kokkos::deep_copy(cellGradients, Scalar(0.0));  // initialize to 0
  int numCells = cellGradients.extent(0);
  Kokkos::parallel_for(numCells, LAMBDA_EXPRESSION(int cellOrdinal) {
    for (int nodeOrdinal = 0; nodeOrdinal < spaceDim;
         nodeOrdinal++)  // "d1" for jacobian
    {
      for (int d = 0; d < spaceDim; d++)  // "d2" for jacobian
      {
        cellGradients(cellOrdinal, nodeOrdinal, d) =
            jacobianInv(cellOrdinal)[nodeOrdinal][d];
        cellGradients(cellOrdinal, spaceDim, d) -=
            jacobianInv(cellOrdinal)[nodeOrdinal][d];
      }
    }
  });

}

template <int spaceDim>
void CellTools::getCellMeasure(
    const FusedJacobianDetView cellMeasure,
    const FusedJacobianDetView jacobianDet) {
  Scalar multiplier = 1.0;
  for (int d = 2; d <= spaceDim; d++) {
    multiplier /= Scalar(d);
  }
  int numCells = jacobianDet.extent(0);
  Kokkos::parallel_for(numCells, LAMBDA_EXPRESSION(int cellOrdinal) {
    cellMeasure(cellOrdinal) =
        std::abs(jacobianDet(cellOrdinal)) * multiplier;
  });
}

void CellTools::mapToPhysicalFrame(
    const PhysPointsView physPoints,
    const RefPointsView  refPoints,
    const PhysPointsView cellWorkset) {
  int numCells = cellWorkset.extent(0);
  //      int numNodes = cellWorkset.extent(1); // should match spaceDim + 1 for the simplex
  int spaceDim = cellWorkset.extent(
      2);  // should match physPoints.extent(2), refPoints.extent(1)

  int numPoints = refPoints.extent(0);

  Kokkos::deep_copy(physPoints, Scalar(0.0));  // initialize to 0
  Kokkos::parallel_for(numCells, LAMBDA_EXPRESSION(int cellOrdinal) {
    for (int ptOrdinal = 0; ptOrdinal < numPoints; ptOrdinal++) {
      int    nodeOrdinal;
      Scalar finalNodeValue = 1.0;
      for (nodeOrdinal = 0; nodeOrdinal < spaceDim; nodeOrdinal++) {
        Scalar nodeValue = refPoints(ptOrdinal, nodeOrdinal);
        finalNodeValue -= nodeValue;
        for (int d = 0; d < spaceDim; d++) {
          physPoints(cellOrdinal, ptOrdinal, d) +=
              nodeValue * cellWorkset(cellOrdinal, nodeOrdinal, d);
        }
      }
      nodeOrdinal = spaceDim;
      for (int d = 0; d < spaceDim; d++) {
        physPoints(cellOrdinal, ptOrdinal, d) +=
            finalNodeValue * cellWorkset(cellOrdinal, nodeOrdinal, d);
      }
    }
  });
}

#define LGR_EXPL_INST(spaceDim) \
template void CellTools::setFusedJacobian<spaceDim>( \
    const FusedJacobianView<spaceDim>, const PhysPointsView); \
template void CellTools::setFusedJacobianDet<spaceDim>( \
    const FusedJacobianDetView, \
    const FusedJacobianView<spaceDim>); \
template void CellTools::setFusedJacobianInv<spaceDim>( \
    const FusedJacobianView<spaceDim>, \
    const FusedJacobianView<spaceDim>); \
template void CellTools::getPhysicalGradients<spaceDim>( \
    const PhysCellGradientView, \
    const FusedJacobianView<spaceDim>); \
template void CellTools::getCellMeasure<spaceDim>( \
    const FusedJacobianDetView, \
    const FusedJacobianDetView);
LGR_EXPL_INST(1)
LGR_EXPL_INST(2)
LGR_EXPL_INST(3)
#undef LGR_EXPL_INST

}  // namespace lgr
