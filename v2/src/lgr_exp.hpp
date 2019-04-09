#ifndef LGR_EXP_HPP
#define LGR_EXP_HPP

#include <Omega_h_scalar.hpp>
#include <lgr_math.hpp>

namespace lgr {

namespace exp {

using Omega_h::Few;
using Omega_h::invert;
using Omega_h::max2;
using Omega_h::identity_tensor;

OMEGA_H_INLINE
double norm_1(Tensor<3> const& A) {
  // 1-norm of matrix is equivalent to max column sum
  // https://en.wikipedia.org/wiki/Matrix_norm
  double v0 = std::abs(A(0, 0) + A(1, 0) + A(2, 0));
  double v1 = std::abs(A(0, 1) + A(1, 1) + A(2, 1));
  double v2 = std::abs(A(0, 2) + A(1, 2) + A(2, 2));
  return max2(max2(v0, v1), v2);
}

//
// Scaling parameter theta for scaling and squaring exponential.
//
OMEGA_H_INLINE
double scaling_squaring_theta(int const order) {
  OMEGA_H_CHECK(order > 0 && order < 22);
  double const theta[] = {
    0.0e-0, 3.7e-8, 5.3e-4, 1.5e-2, 8.5e-2, 2.5e-1, 5.4e-1, 9.5e-1,
    1.5e-0, 2.1e-0, 2.8e-0, 3.6e-0, 4.5e-0, 5.4e-0, 6.3e-0, 7.3e-0,
    8.4e-0, 9,4e-0, 1.1e+1, 1.2e+1, 1.3e+1, 1.4e+1
  };
  return theta[order];
}

//
// Polynomial coefficients for Padé approximants.
//
OMEGA_H_INLINE
double polynomial_coefficient(int const order, int const index) {
  OMEGA_H_CHECK(index <= order);
  double c = 0.0;
  switch (order) {
    case 3:
      {
        double const b[] = {
          120.0, 60.0, 12.0, 1.0};
        c = b[index];
      }
      break;
    case 5:
      {
        double const b[] = {
          30240.0, 15120.0, 3360.0, 420.0, 30.0, 1.0};
        c = b[index];
      }
      break;
    case 7:
      {
        double const b[] = {
          17297280.0, 8648640.0, 1995840.0, 277200.0, 25200.0, 1512.0,
          56.0, 1.0};
        c = b[index];
      }
      break;
    case 9:
      {
        double const b[] = {
          17643225600.0, 8821612800.0, 2075673600.0, 302702400.0,
          30270240.0, 2162160.0, 110880.0, 3960.0, 90.0, 1.0};
        c = b[index];
      }
      break;
    case 13:
      {
        double const b[] = {
          64764752532480000.0, 32382376266240000.0, 7771770303897600.0,
          1187353796428800.0, 129060195264000.0, 10559470521600.0,
          670442572800.0, 33522128640.0, 1323241920.0, 40840800.0,
          960960.0, 16380.0, 182.0, 1.0};
        c = b[index];
      }
      break;
    default:
      bool incorrect_order_in_pade_polynomial = false;
      OMEGA_H_CHECK(incorrect_order_in_pade_polynomial);
  }
  return c;
}

//
// Padé approximant polynomial odd and even terms.
//
OMEGA_H_INLINE
Few<Tensor<3>, 2> pade_polynomial_terms(
    Tensor<3> const& A, int const order) {
  Few<Tensor<3>, 2> out;
  Tensor<3> B = identity_tensor<3>();
  Tensor<3> U = polynomial_coefficient(order, 1) * B;
  Tensor<3> V = polynomial_coefficient(order, 0) * B;
  Tensor<3> const A2 = A * A;
  for (int i = 3; i <= order; i += 2) {
    B = B * A2;
    Tensor<3> const O = polynomial_coefficient(order, i) * B;
    Tensor<3> const E = polynomial_coefficient(order, i - 1) * B;
    U += O;
    V += E;
  }
  U = A * U;
  out[0] = U;
  out[1] = V;
  return out;
}

//
// Compute a non-negative integer power of a tensor by binary manipulation.
//
OMEGA_H_INLINE
Tensor<3> binary_powering(Tensor<3> const& A, int const exponent_in) {
  uint32_t const exponent = static_cast<uint32_t>(exponent_in);
  if (exponent == 0) return identity_tensor<3>();
  uint32_t const rightmost_bit = 1;
  uint32_t const number_digits = 32;
  uint32_t const leftmost_bit = rightmost_bit << (number_digits - 1);
  uint32_t t = 0;
  for (uint32_t j = 0; j < number_digits; ++j) {
    if (((exponent << j) & leftmost_bit) != 0) {
      t = number_digits - j - 1;
      break;
    }
  }
  Tensor<3> P = A;
  uint32_t i = 0;
  uint32_t m = exponent;
  while ((m & rightmost_bit) == 0) {
    P = P * P;
    ++i;
    m = m >> 1;
  }
  Tensor<3> X = P;
  for (uint32_t j = i + 1; j <= t; ++j) {
    P = P * P;
    if (((exponent >> j) & rightmost_bit) != 0) {
      X = X * P;
    }
  }
  return X;
}

//
// Exponential map by squaring and scaling and Padé approximants.
// See algorithm 10.20 in Functions of Matrices, N.J. Higham, SIAM, 2008.
// \param A tensor
// \return \f$ \exp A \f$
//
OMEGA_H_INLINE
Tensor<3> exp(Tensor<3> const& A) {
  Tensor<3> B;
  int const orders[] = {3, 5, 7, 9, 13};
  int const number_orders = 5;
  int const highest_order = orders[number_orders - 1];
  double const norm = norm_1(A);
  for (int i = 0; i < number_orders; ++i) {
    int const order = orders[i];
    double const theta = scaling_squaring_theta(order);
    if (order < highest_order && norm < theta) {
      auto UV = pade_polynomial_terms(A, order);
      auto U = UV[0];
      auto V = UV[1];
      B = invert(V - U) * (U + V);
      break;
    } else if (order == highest_order) {
      double const theta_highest = scaling_squaring_theta(order);
      int const signed_power =
        static_cast<int>(std::ceil(std::log2(norm / theta_highest)));
      int const power_two =
        signed_power > 0 ? static_cast<int>(signed_power) : 0;
      double scale = 1.0;
      for (int j = 0; j < power_two; ++j) {
        scale /= 2.0;
      }
      Tensor<3> const I = identity_tensor<3>();
      Tensor<3> const A1 = scale * A;
      Tensor<3> const A2 = A1 * A1;
      Tensor<3> const A4 = A2 * A2;
      Tensor<3> const A6 = A2 * A4;
      double const b0  = polynomial_coefficient(order, 0);
      double const b1  = polynomial_coefficient(order, 1);
      double const b2  = polynomial_coefficient(order, 2);
      double const b3  = polynomial_coefficient(order, 3);
      double const b4  = polynomial_coefficient(order, 4);
      double const b5  = polynomial_coefficient(order, 5);
      double const b6  = polynomial_coefficient(order, 6);
      double const b7  = polynomial_coefficient(order, 7);
      double const b8  = polynomial_coefficient(order, 8);
      double const b9  = polynomial_coefficient(order, 9);
      double const b10 = polynomial_coefficient(order, 10);
      double const b11 = polynomial_coefficient(order, 11);
      double const b12 = polynomial_coefficient(order, 12);
      double const b13 = polynomial_coefficient(order, 13);
      Tensor<3> const U =
      A1 * ((A6 * (b13 * A6 + b11 * A4 + b9 * A2) +
            b7 * A6 + b5 * A4 + b3 * A2 + b1 * I));
      Tensor<3> const V =
        A6 * (b12 * A6 + b10 * A4 + b8 * A2) +
        b6 * A6 + b4 * A4 + b2 * A2 + b0 * I;
      Tensor<3> const R = invert(V - U) * (U + V);
      int const exponent = (1 << power_two);
      B = binary_powering(R, exponent);
    }
  }
  return B;
}

//
// Exponential map by Taylor series, radius of convergence is infinity
// \param A tensor
// \return \f$ \exp A \f$
//
OMEGA_H_INLINE
Tensor<3> exp_taylor(Tensor<3> const& A) {
  int const max_iter = 128;
  double const tol = DBL_EPSILON;
  Tensor<3> term = identity_tensor<3>();
  // Relative error taken wrt to the first term, which is I and norm = 1
  double relative_error = 1.0;
  Tensor<3> B = term;
  int k = 0;
  while (relative_error > tol && k < max_iter) {
    term = static_cast<double>(1.0 / (k + 1.0)) * term * A;
    B = B + term;
    relative_error = norm_1(term);
    ++k;
  }
  return B;
}

}  // namespace exp

}  // namespace lgr

#endif
