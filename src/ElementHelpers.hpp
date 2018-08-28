#ifndef LGR_ELEMENT_HELPERS_HPP
#define LGR_ELEMENT_HELPERS_HPP
#include "Omega_h_matrix.hpp"
#include "FieldsEnum.hpp"
#include "LGR_Types.hpp"
#include "Fields.hpp"

namespace lgr {

KOKKOS_INLINE_FUNCTION Scalar dot4(const Scalar *a, const Scalar *b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
}
KOKKOS_INLINE_FUNCTION Scalar dot3(const Scalar *a, const Scalar *b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
template <int NumNodes>
KOKKOS_INLINE_FUNCTION Scalar dot(const Scalar *a, const Scalar *b);

template <>
KOKKOS_INLINE_FUNCTION Scalar dot<4>(const Scalar *a, const Scalar *b){
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
}
template <>
KOKKOS_INLINE_FUNCTION Scalar dot<3>(const Scalar *a, const Scalar *b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

KOKKOS_INLINE_FUNCTION Scalar
                       tet4Volume(const Scalar *x, const Scalar *y, const Scalar *z) {
  const Scalar v1[] = {x[1] - x[0], y[1] - y[0], z[1] - z[0]};
  const Scalar v2[] = {x[2] - x[0], y[2] - y[0], z[2] - z[0]};
  const Scalar v3[] = {x[3] - x[0], y[3] - y[0], z[3] - z[0]};
  const Scalar x2 = v2[0];
  const Scalar x3 = v3[0];
  const Scalar y2 = v2[1];
  const Scalar y3 = v3[1];
  const Scalar z2 = v2[2];
  const Scalar z3 = v3[2];
  const Scalar volume =
      (1. / 6.) * (v1[0] * (y2 * z3 - y3 * z2) + v1[1] * (z2 * x3 - z3 * x2) +
                   v1[2] * (x2 * y3 - x3 * y2));
  return volume;
}

/*
  mean-quadrature integrated shape-function gradients
  a.k.a
  the gradients of the nodal basis functions with respect to real coordinates, times the element volume
  a.k.a
  the derivative of the element volume with respect to the position of each node
  a.k.a
  the vector normal to the opposite face times that face's area times two-thirds
  a.k.a
  the cross product of the edge vectors of the opposite face divided by three
*/
KOKKOS_INLINE_FUNCTION void comp_grad(
    const Scalar *const x,
    const Scalar *const y,
    const Scalar *const z,
    Scalar *const  grad_x,
    Scalar *const  grad_y,
    Scalar *const  grad_z) {
  constexpr Scalar r6 = 1.0 / 6.0;

  grad_x[0] =
      r6 * (y[1] * (z[3] - z[2]) + y[2] * (z[1] - z[3]) + y[3] * (z[2] - z[1]));
  grad_x[1] =
      r6 * (y[2] * (z[3] - z[0]) + y[0] * (z[2] - z[3]) + y[3] * (z[0] - z[2]));
  grad_x[2] =
      r6 * (y[0] * (z[3] - z[1]) + y[1] * (z[0] - z[3]) + y[3] * (z[1] - z[0]));
  grad_x[3] =
      r6 * (y[0] * (z[1] - z[2]) + y[2] * (z[0] - z[1]) + y[1] * (z[2] - z[0]));

  grad_y[0] =
      r6 * (z[1] * (x[3] - x[2]) + z[2] * (x[1] - x[3]) + z[3] * (x[2] - x[1]));
  grad_y[1] =
      r6 * (z[2] * (x[3] - x[0]) + z[0] * (x[2] - x[3]) + z[3] * (x[0] - x[2]));
  grad_y[2] =
      r6 * (z[0] * (x[3] - x[1]) + z[1] * (x[0] - x[3]) + z[3] * (x[1] - x[0]));
  grad_y[3] =
      r6 * (z[0] * (x[1] - x[2]) + z[2] * (x[0] - x[1]) + z[1] * (x[2] - x[0]));

  grad_z[0] =
      r6 * (x[1] * (y[3] - y[2]) + x[2] * (y[1] - y[3]) + x[3] * (y[2] - y[1]));
  grad_z[1] =
      r6 * (x[2] * (y[3] - y[0]) + x[0] * (y[2] - y[3]) + x[3] * (y[0] - y[2]));
  grad_z[2] =
      r6 * (x[0] * (y[3] - y[1]) + x[1] * (y[0] - y[3]) + x[3] * (y[1] - y[0]));
  grad_z[3] =
      r6 * (x[0] * (y[1] - y[2]) + x[2] * (y[0] - y[1]) + x[1] * (y[2] - y[0]));
}

KOKKOS_INLINE_FUNCTION
Omega_h::Few< Omega_h::Vector<3>, 4>
comp_reference_nodal_gradients(const Scalar *const /*xi*/)
{
  typedef Omega_h::Vector<3> Vector;
  typedef Omega_h::Few<Vector,4> Bucket;
  Bucket returnMe;
  {
    Vector &GRAD = returnMe[0]; 
    GRAD(0) = -1.;
    GRAD(1) = -1.;
    GRAD(2) = -1.;
  }
  {
    Vector &GRAD = returnMe[1]; 
    GRAD(0) = +1.;
    GRAD(1) = +0.;
    GRAD(2) = +0.;
  }
  {
    Vector &GRAD = returnMe[2]; 
    GRAD(0) = +0.;
    GRAD(1) = +1.;
    GRAD(2) = +0.;
  }
  {
    Vector &GRAD = returnMe[3]; 
    GRAD(0) = +0.;
    GRAD(1) = +0.;
    GRAD(2) = +1.;
  }
  return returnMe;
}

KOKKOS_INLINE_FUNCTION 
Omega_h::Few< Omega_h::Vector<3>, 4> 
comp_face_basis( const Scalar *const x,
		 const Scalar *const y,
		 const Scalar *const z,
		 const Scalar *const xi ) 
{
  typedef Omega_h::Vector<3> Vector;
  typedef Omega_h::Matrix<3,3> Tensor;
  typedef Omega_h::Few<Vector,4> Bucket;

  const Bucket gradN = comp_reference_nodal_gradients(xi);
  Tensor JinvF = Omega_h::zero_matrix<3,3>();
  for (int A=0; A<4; ++A) {
    Vector xA;
    xA(0) = x[A];
    xA(1) = y[A];
    xA(2) = z[A];
    Vector gradA = gradN[A];
    const Tensor addMe = Omega_h::outer_product(xA,gradA);
    JinvF += addMe;
  }
  const Scalar J = Omega_h::determinant(JinvF);
  JinvF /= J;
 
  /*
    intrepid face 0
    outputValues(0, i0, 0) = 2.0*x;
    outputValues(0, i0, 1) = 2.0*(y - 1.0);
    outputValues(0, i0, 2) = 2.0*z;

    intreped face 1
    outputValues(1, i0, 0) = 2.0*x;
    outputValues(1, i0, 1) = 2.0*y;
    outputValues(1, i0, 2) = 2.0*z;
    
    intreped face 2
    outputValues(2, i0, 0) = 2.0*(x - 1.0);
    outputValues(2, i0, 1) = 2.0*y;
    outputValues(2, i0, 2) = 2.0*z;
    
    intrepid face 3
    outputValues(3, i0, 0) = 2.0*x;
    outputValues(3, i0, 1) = 2.0*y;
    outputValues(3, i0, 2) = 2.0*(z - 1.0);
  */

  Bucket returnMe;
  {
    //Omega_h face 0 = intrepid face +3
    Vector &fillMe = returnMe[0];
    fillMe(0) = 2.0*xi[0];
    fillMe(1) = 2.0*xi[1];
    fillMe(2) = 2.0*(xi[2] - 1.0);
    fillMe = JinvF*fillMe;
  }
  {
    //Omega_h face 1 = intrepid face +0
    Vector &fillMe = returnMe[1];
    fillMe(0) = 2.0*xi[0];
    fillMe(1) = 2.0*(xi[1] - 1.0);
    fillMe(2) = 2.0*xi[2];
    fillMe = JinvF*fillMe;
  }
  {
    //Omega_h face 2 = intrepid face +1
    Vector &fillMe = returnMe[2];
    fillMe(0) = 2.0*xi[0];
    fillMe(1) = 2.0*xi[1];
    fillMe(2) = 2.0*xi[2];
    fillMe = JinvF*fillMe;
  }
  {
    //Omega_h face 3 = intrepid face +2
    Vector &fillMe = returnMe[3];
    fillMe(0) = 2.0*(xi[0] - 1.0);
    fillMe(1) = 2.0*xi[1];
    fillMe(2) = 2.0*xi[2];
    fillMe = JinvF*fillMe;
  }
  
  return returnMe;
}


KOKKOS_INLINE_FUNCTION Scalar
                       maxEdgeLength(const Scalar *x, const Scalar *y, const Scalar *z) {
  Scalar    h = 0.0;
  const int node0[] = {0, 1, 2, 0, 1, 2};
  const int node1[] = {1, 2, 0, 3, 3, 3};
  for (int e = 0; e < 6; ++e) {
    const int    n0 = node0[e];
    const int    n1 = node1[e];
    const Scalar dx = x[n1] - x[n0];
    const Scalar dy = y[n1] - y[n0];
    const Scalar dz = z[n1] - z[n0];
    const Scalar he = sqrt(dx * dx + dy * dy + dz * dz);
    h = Omega_h::max2(h, he);
  }
  return h;
}

KOKKOS_INLINE_FUNCTION Scalar
                       minEdgeLength(const Scalar *x, const Scalar *y, const Scalar *z) {
  Scalar    h = Omega_h::ArithTraits<Scalar>::max();
  const int node0[] = {0, 1, 2, 0, 1, 2};
  const int node1[] = {1, 2, 0, 3, 3, 3};
  for (int e = 0; e < 6; ++e) {
    const int    n0 = node0[e];
    const int    n1 = node1[e];
    const Scalar dx = x[n1] - x[n0];
    const Scalar dy = y[n1] - y[n0];
    const Scalar dz = z[n1] - z[n0];
    const Scalar he = sqrt(dx * dx + dy * dy + dz * dz);
    h = Omega_h::min2(h, he);
  }
  return h;
}

template <int SpatialDim>
KOKKOS_INLINE_FUNCTION void deviatoricProjection(
    Scalar sigma
        [lgr::Fields<SpatialDim>::SymTensorLength]) {
  Scalar vol = 0.0;
  for (int d = 0; d < SpatialDim; d++) {
    vol += sigma[SymTensorDiagonalIndices<SpatialDim>(d)];
  }
  vol /= Scalar(SpatialDim);
  for (int d = 0; d < SpatialDim; d++) {
    sigma[SymTensorDiagonalIndices<SpatialDim>(d)] -= vol;
  }
}

}

#endif
