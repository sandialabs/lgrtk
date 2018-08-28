//
//  CellTools_Simplex.hpp
//
//
//  Created by Roberts, Nathan V on 6/20/17.
//
//

#ifndef LGR_CELLTOOLS_SIMPLEX_HPP
#define LGR_CELLTOOLS_SIMPLEX_HPP

#include <Kokkos_Core.hpp>
#include <Omega_h_matrix.hpp>
#include "LGR_Types.hpp"

namespace lgr {

class CellTools {
 public:
  typedef Kokkos::View<Scalar****, Layout, MemSpace> JacobianView;  // (C,P,D,D)
  typedef Kokkos::View<Scalar**, Layout, MemSpace>   JacobianDetView;  // (C,P)
  typedef Kokkos::View<Scalar***, Layout, MemSpace>
                                                    PhysPointsView;  // (C,P,D) (For cellWorksets, P = N, # of nodes)
  typedef Kokkos::View<Scalar**, Layout, MemSpace>  RefPointsView;    // (P,D)
  typedef Kokkos::View<Scalar***, Layout, MemSpace> RefGradientView;  // (N,P,D)

  template <int spaceDim>
  using FusedJacobianView = Kokkos::View<
      Omega_h::Matrix<spaceDim, spaceDim>*,
      Layout,
      MemSpace>;  // (C) -- with simplices, we know the Jacobian is independent of the point.

  typedef Kokkos::View<Scalar*, Layout, MemSpace> FusedJacobianDetView;  // (C)
  typedef Kokkos::View<Scalar***, Layout, MemSpace>
      PhysCellGradientView;  // (C,F,D) -- F == N, the number of basis elements

  // workspace argument should be (N,P,D): this gives us storage that we don't have to reallocate internally each time we are called
  static void setJacobian(
      const JacobianView    jacobian,
      const RefPointsView   points,
      const PhysPointsView  cellWorkset,
      const RefGradientView workspace);

  // this is our first effort at a point-independent Jacobian computation
  template <int spaceDim>
  static void setFusedJacobian(
      const FusedJacobianView<spaceDim> jacobian,
      const PhysPointsView              cellWorkset);

  static void setJacobianDet(
      const JacobianDetView jacobianDet, const JacobianView jacobian);

  template <int spaceDim>
  static void setFusedJacobianDet(
      const FusedJacobianDetView        jacobianDet,
      const FusedJacobianView<spaceDim> jacobian);

  // TODO: write a unit test against this
  template <int spaceDim>
  static void setFusedJacobianInv(
      const FusedJacobianView<spaceDim> jacobianInv,
      const FusedJacobianView<spaceDim> jacobian);

  // TODO: write a unit test against this
  template <int spaceDim>
  static void getPhysicalGradients(
      const PhysCellGradientView        cellGradients,
      const FusedJacobianView<spaceDim> jacobianInv);

  template <int spaceDim>
  static void getCellMeasure(
      const FusedJacobianDetView cellMeasure,
      const FusedJacobianDetView jacobianDet);

  static void mapToPhysicalFrame(
      const PhysPointsView physPoints,
      const RefPointsView  refPoints,
      const PhysPointsView cellWorkset);

};

#define LGR_EXPL_INST_DECL(spaceDim) \
extern template void CellTools::setFusedJacobian<spaceDim>( \
    const FusedJacobianView<spaceDim>, const CellTools::PhysPointsView); \
extern template void CellTools::setFusedJacobianDet<spaceDim>( \
    const FusedJacobianDetView, \
    const FusedJacobianView<spaceDim>); \
extern template void CellTools::setFusedJacobianInv<spaceDim>( \
    const FusedJacobianView<spaceDim>, \
    const FusedJacobianView<spaceDim>); \
extern template void CellTools::getPhysicalGradients<spaceDim>( \
    const PhysCellGradientView, \
    const FusedJacobianView<spaceDim>); \
extern template void CellTools::getCellMeasure<spaceDim>( \
    const FusedJacobianDetView, \
    const FusedJacobianDetView);
LGR_EXPL_INST_DECL(1)
LGR_EXPL_INST_DECL(2)
LGR_EXPL_INST_DECL(3)
#undef LGR_EXPL_INST_DECL

}  // namespace lgr

#endif /* CellTools_Simplex_h */
