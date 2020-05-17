#pragma once

#include <tuple>

#include <hpc_vector3.hpp>
#include <hpc_array_traits.hpp>
#include <hpc_tensor_detail.hpp>
#include <hpc_functional.hpp>

namespace hpc {

template <typename Scalar>
class matrix3x3 {
public:
  using scalar_type = Scalar;
private:
  scalar_type raw[3][3];
public:
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr matrix3x3(
      Scalar const a, Scalar const b, Scalar const c,
      Scalar const d, Scalar const e, Scalar const f,
      Scalar const g, Scalar const h, Scalar const i) noexcept
  {
    raw[0][0] = a;
    raw[0][1] = b;
    raw[0][2] = c;
    raw[1][0] = d;
    raw[1][1] = e;
    raw[1][2] = f;
    raw[2][0] = g;
    raw[2][1] = h;
    raw[2][2] = i;
  }
//  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr matrix3x3(symmetric3x3<Scalar> const& A) noexcept
//  {
//    matrix3x3(A(0,0), A(0,1), A(0,2), A(1,0), A(1,1), A(1,2), A(2,0), A(2,1), A(2,2));
//  }
  HPC_ALWAYS_INLINE matrix3x3() noexcept = default;
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE static constexpr matrix3x3 identity() noexcept {
    return matrix3x3(
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0);
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE static constexpr matrix3x3 zero() noexcept {
    return matrix3x3(
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0);
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr Scalar operator()(int const i, int const j) const noexcept {
    return raw[i][j];
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE Scalar& operator()(int const i, int const j) noexcept {
    return raw[i][j];
  }
};

template <class T>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr auto
operator+(matrix3x3<T> left, matrix3x3<T> right) noexcept {
  return matrix3x3<T>(
      left(0, 0) + right(0, 0),
      left(0, 1) + right(0, 1),
      left(0, 2) + right(0, 2),
      left(1, 0) + right(1, 0),
      left(1, 1) + right(1, 1),
      left(1, 2) + right(1, 2),
      left(2, 0) + right(2, 0),
      left(2, 1) + right(2, 1),
      left(2, 2) + right(2, 2));
}

template <class T>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE matrix3x3<T>&
operator+=(matrix3x3<T>& left, matrix3x3<T> right) noexcept {
  left = left + right;
  return left;
}

template <class L, class R>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr auto
operator+(matrix3x3<L> left, R right) noexcept {
  return matrix3x3<L>(
      left(0, 0) + right,
      left(0, 1),
      left(0, 2),
      left(1, 0),
      left(1, 1) + right,
      left(1, 2),
      left(2, 0),
      left(2, 1),
      left(2, 2) + right);
}

template <class L, class R>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE matrix3x3<L>&
operator+=(matrix3x3<L>& left, R right) noexcept {
  left = left + right;
  return left;
}

template <class L, class R>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr auto
operator+(L left, matrix3x3<R> right) noexcept {
  return right + left;
}

template <class T>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr auto
operator-(matrix3x3<T> left, matrix3x3<T> right) noexcept {
  return matrix3x3<T>(
      left(0, 0) - right(0, 0),
      left(0, 1) - right(0, 1),
      left(0, 2) - right(0, 2),
      left(1, 0) - right(1, 0),
      left(1, 1) - right(1, 1),
      left(1, 2) - right(1, 2),
      left(2, 0) - right(2, 0),
      left(2, 1) - right(2, 1),
      left(2, 2) - right(2, 2));
}

template <class T>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE matrix3x3<T>&
operator-=(matrix3x3<T>& left, matrix3x3<T> right) noexcept {
  left = left - right;
  return left;
}

template <class L, class R>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr auto
operator-(matrix3x3<L> left, R right) noexcept {
  return matrix3x3<L>(
      left(0, 0) - right.raw,
      left(0, 1),
      left(0, 2),
      left(1, 0),
      left(1, 1) - right.raw,
      left(1, 2),
      left(2, 0),
      left(2, 1),
      left(2, 2) - right.raw);
}

template <class L, class R>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE matrix3x3<L>&
operator-=(matrix3x3<L>& left, R const right) noexcept {
  left = left - right;
  return left;
}

template <class L, class R>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr auto
operator*(matrix3x3<L> left, matrix3x3<R> right) noexcept {
  return matrix3x3<decltype(L() * R())>(
      left(0, 0) * right(0, 0) + left(0, 1) * right(1, 0) + left(0, 2) * right(2, 0),
      left(0, 0) * right(0, 1) + left(0, 1) * right(1, 1) + left(0, 2) * right(2, 1),
      left(0, 0) * right(0, 2) + left(0, 1) * right(1, 2) + left(0, 2) * right(2, 2),
      left(1, 0) * right(0, 0) + left(1, 1) * right(1, 0) + left(1, 2) * right(2, 0),
      left(1, 0) * right(0, 1) + left(1, 1) * right(1, 1) + left(1, 2) * right(2, 1),
      left(1, 0) * right(0, 2) + left(1, 1) * right(1, 2) + left(1, 2) * right(2, 2),
      left(2, 0) * right(0, 0) + left(2, 1) * right(1, 0) + left(2, 2) * right(2, 0),
      left(2, 0) * right(0, 1) + left(2, 1) * right(1, 1) + left(2, 2) * right(2, 1),
      left(2, 0) * right(0, 2) + left(2, 1) * right(1, 2) + left(2, 2) * right(2, 2));
}

template <class L, class R>
HPC_HOST_DEVICE matrix3x3<L>&
operator*=(matrix3x3<L>& left, matrix3x3<R> right) noexcept {
  left = left * right;
  return left;
}

template <class L, class R>
HPC_HOST_DEVICE constexpr auto
operator*(matrix3x3<L> left, vector3<R> right) noexcept {
  return vector3<decltype(L() * R())>(
      left(0, 0) * right(0) + left(0, 1) * right(1) + left(0, 2) * right(2),
      left(1, 0) * right(0) + left(1, 1) * right(1) + left(1, 2) * right(2),
      left(2, 0) * right(0) + left(2, 1) * right(1) + left(2, 2) * right(2));
}

template <class L, class R>
HPC_HOST_DEVICE constexpr auto
operator*(vector3<L> left, matrix3x3<R> right) noexcept {
  return vector3<decltype(L() * R())>(
      left(0) * right(0, 0) + left(1) * right(1, 0) + left(2) * right(2, 0),
      left(0) * right(0, 1) + left(1) * right(1, 1) + left(2) * right(2, 1),
      left(0) * right(0, 2) + left(1) * right(1, 2) + left(2) * right(2, 2));
}

template <class L, class R>
HPC_HOST_DEVICE constexpr auto
operator*(matrix3x3<L> left, R right) noexcept {
  return matrix3x3<decltype(L() * R())>(
      left(0, 0) * right, left(0, 1) * right, left(0, 2) * right,
      left(1, 0) * right, left(1, 1) * right, left(1, 2) * right,
      left(2, 0) * right, left(2, 1) * right, left(2, 2) * right);
}

template <class L, class R>
HPC_HOST_DEVICE matrix3x3<L>&
operator*=(matrix3x3<L>& left, R right) noexcept {
  left = left * right;
  return left;
}

template <class L, class R>
HPC_HOST_DEVICE constexpr auto
operator*(L left, matrix3x3<R> right) noexcept {
  return right * left;
}

template <class L, class R>
HPC_HOST_DEVICE constexpr auto
operator/(matrix3x3<L> left, R right) noexcept {
  return matrix3x3<decltype(L() / R())>(
      left(0, 0) / right, left(0, 1) / right, left(0, 2) / right,
      left(1, 0) / right, left(1, 1) / right, left(1, 2) / right,
      left(2, 0) / right, left(2, 1) / right, left(2, 2) / right);
}

template <class L, class R>
HPC_HOST_DEVICE matrix3x3<L>&
operator/=(matrix3x3<L>& left, R right) noexcept {
  left = left / right;
  return left;
}

template <class L, class R>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr auto
inner_product(matrix3x3<L> const left, matrix3x3<R> const right) noexcept {
  return (
    left(0, 0)*right(0, 0) + left(0, 1)*right(0, 1) + left(0, 2)*right(0, 2) +
    left(1, 0)*right(1, 0) + left(1, 1)*right(1, 1) + left(1, 2)*right(1, 2) +
    left(2, 0)*right(2, 0) + left(2, 1)*right(2, 1) + left(2, 2)*right(2, 2)
  );
}

template <class T>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr T
norm(matrix3x3<T> const x) noexcept {
  return std::sqrt(inner_product(x, x));
}

// \return \f$ \max_{j \in {0,\cdots,N}}\Sigma_{i=0}^N |A_{ij}| \f$
template <class T>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr T
norm_1(matrix3x3<T> const A) noexcept {
  auto const v0 = std::abs(A(0,0)) + std::abs(A(1,0)) + std::abs(A(2,0));
  auto const v1 = std::abs(A(0,1)) + std::abs(A(1,1)) + std::abs(A(2,1));
  auto const v2 = std::abs(A(0,2)) + std::abs(A(1,2)) + std::abs(A(2,2));
  return hpc::max(hpc::max(v0, v1), v2);
}

// \return \f$ \max_{i \in {0,\cdots,N}}\Sigma_{j=0}^N |A_{ij}| \f$
template <class T>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr T
norm_infinity(matrix3x3<T> const A) noexcept {
  auto const v0 = std::abs(A(0,0)) + std::abs(A(0,1)) + std::abs(A(0,2));
  auto const v1 = std::abs(A(1,0)) + std::abs(A(1,1)) + std::abs(A(1,2));
  auto const v2 = std::abs(A(2,0)) + std::abs(A(2,1)) + std::abs(A(2,2));
  return hpc::max(hpc::max(v0, v1), v2);
}

template <class T>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr matrix3x3<T>
transpose(matrix3x3<T> x) noexcept {
  return matrix3x3<T>(
      x(0, 0),
      x(1, 0),
      x(2, 0),
      x(0, 1),
      x(1, 1),
      x(2, 1),
      x(0, 2),
      x(1, 2),
      x(2, 2));
}

template <class L, class R>
HPC_HOST_DEVICE constexpr auto
outer_product(vector3<L> left, vector3<R> right) noexcept {
  return matrix3x3<decltype(L() * R())>(
      left(0) * right(0),
      left(0) * right(1),
      left(0) * right(2),
      left(1) * right(0),
      left(1) * right(1),
      left(1) * right(2),
      left(2) * right(0),
      left(2) * right(1),
      left(2) * right(2));
}

template <typename Scalar>
HPC_HOST_DEVICE constexpr auto
determinant(matrix3x3<Scalar> const x) noexcept {
  Scalar const a = x(0, 0);
  Scalar const b = x(0, 1);
  Scalar const c = x(0, 2);
  Scalar const d = x(1, 0);
  Scalar const e = x(1, 1);
  Scalar const f = x(1, 2);
  Scalar const g = x(2, 0);
  Scalar const h = x(2, 1);
  Scalar const i = x(2, 2);
  return (a * e * i) + (b * f * g) + (c * d * h) - (c * e * g) - (b * d * i) -
         (a * f * h);
}

template <class T>
HPC_HOST_DEVICE constexpr auto
inverse_fast(matrix3x3<T> const x) {
  auto const a = x(0, 0);
  auto const b = x(0, 1);
  auto const c = x(0, 2);
  auto const d = x(1, 0);
  auto const e = x(1, 1);
  auto const f = x(1, 2);
  auto const g = x(2, 0);
  auto const h = x(2, 1);
  auto const i = x(2, 2);
  auto const A = (e * i - f * h);
  auto const D = -(b * i - c * h);
  auto const G = (b * f - c * e);
  auto const B = -(d * i - f * g);
  auto const E = (a * i - c * g);
  auto const H = -(a * f - c * d);
  auto const C = (d * h - e * g);
  auto const F = -(a * h - b * g);
  auto const I = (a * e - b * d);
  using num_t    = matrix3x3<std::remove_const_t<decltype(A)>>;
  auto const num = num_t(A, D, G, B, E, H, C, F, I);
  return num / determinant(x);
}

template <typename Scalar>
HPC_HOST_DEVICE constexpr auto
det(matrix3x3<Scalar> const A) noexcept {
  return determinant(A);
}

template <class T>
HPC_HOST_DEVICE constexpr auto
inverse(matrix3x3<T> const x) {
  return inverse_fast(x);
}

// Logarithm by Gregory series. Convergence guaranteed for symmetric A
template <typename T>
HPC_HOST constexpr auto
log_gregory(matrix3x3<T> const& A)
{
  auto const max_iter  = 8192;
  auto const tol       = machine_epsilon<T>();
  auto const I         = matrix3x3<T>::identity();
  auto const IpA       = I + A;
  auto const ImA       = I - A;
  auto       S         = ImA * inverse(IpA);
  auto       norm_s    = norm(S);
  auto const C         = S * S;
  auto       B         = S;
  auto       k         = 0;
  while (norm_s > tol && ++k <= max_iter) {
    S = (2.0 * k - 1.0) * S * C / (2.0 * k + 1.0);
    B += S;
    norm_s    = norm(S);
  }
  B *= -2.0;
  return B;
}

// Inverse by full pivot. Since this is 3x3, can afford it, and avoids
// cancellation errors as much as possible. This is important for an
// explicit dynamics code that will perform a huge number of these
// calculations.
template <typename T>
HPC_HOST_DEVICE constexpr auto
inverse_full_pivot(matrix3x3<T> const& A)
{
  auto S = A;
  auto B = matrix3x3<T>::identity();
  unsigned int intact_rows = (1U << 3) - 1;
  unsigned int intact_cols = intact_rows;
  // Gauss-Jordan elimination with full pivoting
  for (auto k = 0; k < 3; ++k) {
    // Determine full pivot
    auto pivot = 0.0;
    auto pivot_row = 3;
    auto pivot_col = 3;
    for (auto row = 0; row < 3; ++row) {
      if (!(intact_rows & (1 << row))) continue;
      for (auto col = 0; col < 3; ++col) {
        if (!(intact_cols & (1 << col))) continue;
        auto s = std::abs(S(row, col));
        if (s > pivot) {
          pivot_row = row;
          pivot_col = col;
          pivot = s;
        }
      }
    }
    assert(pivot_row < 3);
    assert(pivot_col < 3);
    // Gauss-Jordan elimination
    auto const t = S(pivot_row, pivot_col);
    assert(t != 0.0);
    for (auto j = 0; j < 3; ++j) {
      S(pivot_row, j) /= t;
      B(pivot_row, j) /= t;
    }

    for (auto i = 0; i < 3; ++i) {
      if (i == pivot_row) continue;
      auto const c = S(i, pivot_col);
      for (auto j = 0; j < 3; ++j) {
        S(i, j) -= c * S(pivot_row, j);
        B(i, j) -= c * B(pivot_row, j);
      }
    }
    // Eliminate current row and col from intact rows and cols
    intact_rows &= ~(1 << pivot_row);
    intact_cols &= ~(1 << pivot_col);
  }
  return transpose(S) * B;
}

// Solve by full pivot. Since this is 3x3, can afford it, and avoids
// cancellation errors as much as possible. This is important for an
// explicit dynamics code that will perform a huge number of these
// calculations.
template <typename T>
HPC_HOST_DEVICE constexpr auto
solve_full_pivot(matrix3x3<T> const& A, vector3<T> const& b)
{
  auto S = A;
  auto B = b;
  unsigned int intact_rows = (1U << 3) - 1;
  unsigned int intact_cols = intact_rows;
  // Gauss-Jordan elimination with full pivoting
  for (auto k = 0; k < 3; ++k) {
    // Determine full pivot
    auto pivot = 0.0;
    auto pivot_row = 3;
    auto pivot_col = 3;
    for (auto row = 0; row < 3; ++row) {
      if (!(intact_rows & (1 << row))) continue;
      for (auto col = 0; col < 3; ++col) {
        if (!(intact_cols & (1 << col))) continue;
        auto s = std::abs(S(row, col));
        if (s > pivot) {
          pivot_row = row;
          pivot_col = col;
          pivot = s;
        }
      }
    }
    assert(pivot_row < 3);
    assert(pivot_col < 3);
    // Gauss-Jordan elimination
    auto const t = S(pivot_row, pivot_col);
    assert(t != 0.0);
    for (auto j = 0; j < 3; ++j) {
      S(pivot_row, j) /= t;
    }
    B(pivot_row) /= t;

    for (auto i = 0; i < 3; ++i) {
      if (i == pivot_row) continue;
      auto const c = S(i, pivot_col);
      for (auto j = 0; j < 3; ++j) {
        S(i, j) -= c * S(pivot_row, j);
      }
      B(i) -= c * B(pivot_row);
    }
    // Eliminate current row and col from intact rows and cols
    intact_rows &= ~(1 << pivot_row);
    intact_cols &= ~(1 << pivot_col);
  }
  return transpose(S) * B;
}

// Matrix square root by product form of Denman-Beavers iteration.
template <typename T>
HPC_HOST_DEVICE constexpr auto
sqrt_dbp(matrix3x3<T> const& A, int& k)
{
  auto const eps = machine_epsilon<T>();
  auto const tol = 0.5 * std::sqrt(3.0) * eps; // 3 is dim
  auto const I = matrix3x3<T>::identity();
  auto const max_iter = 32;
  auto X = A;
  auto M = A;
  auto scale = true;
  k = 0;
  while (k++ < max_iter) {
    if (scale == true) {
      auto const d = std::abs(det(M));
      auto const d2 = std::sqrt(d);
      auto const d6 = std::cbrt(d2);
      auto const g = 1.0 / d6;
      X *= g;
      M *= g * g;
    }
    auto const Y = X;
    auto const N = inverse(M);
    X *= 0.5 * (I + N);
    M = 0.5 * (I + 0.5 * (M + N));
    auto const error = norm(M - I);
    auto const diff = norm(X - Y) / norm(X);
    scale = diff >= 0.01;
    if (error <= tol) break;
  }
  return X;
}

// Matrix square root
template <typename T>
HPC_HOST_DEVICE constexpr auto
sqrt(matrix3x3<T> const& A)
{
  int i = 0;
  return sqrt_dbp(A, i);
}

// Logarithmic map by Padé approximant and partial fractions
template <typename T>
HPC_HOST_DEVICE constexpr auto
log_pade_pf(matrix3x3<T> const& A, int const n)
{
  auto const I = matrix3x3<T>::identity();
  auto X = 0.0 * A;
  for (auto i = 0; i < n; ++i) {
    auto const x = 0.5 * (1.0 + gauss_legendre_abscissae<T>(n, i));
    auto const w = 0.5 * gauss_legendre_weights<T>(n, i);
    auto const B = I + x * A;
    X += w * A * inverse_full_pivot(B);
  }
  return X;
}

// Logarithmic map by inverse scaling and squaring and Padé approximants
template <typename T>
HPC_HOST_DEVICE constexpr auto
log_iss(matrix3x3<T> const& A)
{
  auto const I = matrix3x3<T>::identity();
  auto const c15 = pade_coefficients<T>(15);
  auto X = A;
  auto i = 5;
  auto j = 0;
  auto k = 0;
  auto m = 0;
  while (true) {
    auto const diff = norm_1(X - I);
    if (diff <= c15) {
      auto p = 2; while(pade_coefficients<T>(p) <= diff && p < 16) {++p;}
      auto q = 2; while(pade_coefficients<T>(q) <= diff / 2.0 && q < 16) {++q;}
      if ((2 * (p - q) / 3) < i || ++j == 2) {m = p + 1; break;}
    }
    X = sqrt_dbp(X, i); ++k;
  }
  X = (1U << k) * log_pade_pf(X - I, m);
  return X;
}

// Logarithmic map
template <typename T>
HPC_HOST_DEVICE constexpr auto
log(matrix3x3<T> const& A)
{
  return log_iss(A);
}

template <typename T>
HPC_HOST_DEVICE constexpr auto
pade_polynomial_terms(matrix3x3<T> const& A, int const order,
    matrix3x3<T>& U, matrix3x3<T>& V) {
  auto B = matrix3x3<T>::identity();
  U = polynomial_coefficient<T>(order, 1) * B;
  V = polynomial_coefficient<T>(order, 0) * B;
  auto const A2 = A * A;
  for (int i = 3; i <= order; i += 2) {
    B = B * A2;
    auto const O = polynomial_coefficient<T>(order, i) * B;
    auto const E = polynomial_coefficient<T>(order, i - 1) * B;
    U += O;
    V += E;
  }
  U = A * U;
}

// Compute a non-negative integer power of a tensor by binary manipulation.
template <typename T>
HPC_HOST_DEVICE constexpr auto
binary_powering(matrix3x3<T> const& A, int const e) {
  using bits = uint64_t;
  bits const number_digits = 64;
  bits const exponent = static_cast<bits>(e);
  if (exponent == 0) return matrix3x3<T>::identity();
  bits const rightmost_bit = 1;
  bits const leftmost_bit = rightmost_bit << (number_digits - 1);
  bits t = 0;
  for (bits j = 0; j < number_digits; ++j) {
    if (((exponent << j) & leftmost_bit) != 0) {
      t = number_digits - j - 1;
      break;
    }
  }
  auto P = A;
  bits i = 0;
  bits m = exponent;
  while ((m & rightmost_bit) == 0) {
    P = P * P;
    ++i;
    m = m >> 1;
  }
  auto X = P;
  for (bits j = i + 1; j <= t; ++j) {
    P = P * P;
    if (((exponent >> j) & rightmost_bit) != 0) {
      X = X * P;
    }
  }
  return X;
}

// Exponential map by squaring and scaling and Padé approximants.
// See algorithm 10.20 in Functions of Matrices, N.J. Higham, SIAM, 2008.
template <typename T>
HPC_HOST_DEVICE constexpr auto
exp(matrix3x3<T> const& A) {
  auto B = matrix3x3<T>::identity();
  int const orders[] = {3, 5, 7, 9, 13};
  auto const number_orders = 5;
  auto const highest_order = orders[number_orders - 1];
  auto const norm = norm_1(A);
  for (auto i = 0; i < number_orders; ++i) {
    auto const order = orders[i];
    auto const theta = scaling_squaring_theta<T>(order);
    if (order < highest_order && norm < theta) {
      auto U = B;
      auto V = B;
      pade_polynomial_terms(A, order, U, V);
      B = inverse(V - U) * (U + V);
      break;
    } else if (order == highest_order) {
      auto const theta_highest = scaling_squaring_theta<T>(order);
      auto const signed_power =
        static_cast<int>(std::ceil(std::log2(norm / theta_highest)));
      auto const power_two =
        signed_power > 0 ? static_cast<int>(signed_power) : 0;
      auto scale = 1.0;
      for (int j = 0; j < power_two; ++j) {
        scale /= 2.0;
      }
      auto const I = matrix3x3<T>::identity();
      auto const A1 = scale * A;
      auto const A2 = A1 * A1;
      auto const A4 = A2 * A2;
      auto const A6 = A2 * A4;
      auto const b0  = polynomial_coefficient<T>(order, 0);
      auto const b1  = polynomial_coefficient<T>(order, 1);
      auto const b2  = polynomial_coefficient<T>(order, 2);
      auto const b3  = polynomial_coefficient<T>(order, 3);
      auto const b4  = polynomial_coefficient<T>(order, 4);
      auto const b5  = polynomial_coefficient<T>(order, 5);
      auto const b6  = polynomial_coefficient<T>(order, 6);
      auto const b7  = polynomial_coefficient<T>(order, 7);
      auto const b8  = polynomial_coefficient<T>(order, 8);
      auto const b9  = polynomial_coefficient<T>(order, 9);
      auto const b10 = polynomial_coefficient<T>(order, 10);
      auto const b11 = polynomial_coefficient<T>(order, 11);
      auto const b12 = polynomial_coefficient<T>(order, 12);
      auto const b13 = polynomial_coefficient<T>(order, 13);
      auto const U =
      A1 * ((A6 * (b13 * A6 + b11 * A4 + b9 * A2) +
            b7 * A6 + b5 * A4 + b3 * A2 + b1 * I));
      auto const V =
        A6 * (b12 * A6 + b10 * A4 + b8 * A2) +
        b6 * A6 + b4 * A4 + b2 * A2 + b0 * I;
      auto const R = inverse(V - U) * (U + V);
      auto const exponent = (1 << power_two);
      B = binary_powering(R, exponent);
    }
  }
  return B;
}

// Exponential map by Taylor series, radius of convergence is infinity
template <typename T>
HPC_HOST_DEVICE constexpr auto
exp_taylor(matrix3x3<T> const& A) {
  auto const max_iter = 1024;
  auto const tol = machine_epsilon<T>();
  auto term = matrix3x3<T>::identity();
  // Relative error taken wrt to the first term, which is I and norm = 1
  auto relative_error = 1.0;
  auto B = term;
  auto k = 0;
  while (relative_error > tol && k < max_iter) {
    term = static_cast<T>(1.0 / (k + 1.0)) * term * A;
    B = B + term;
    relative_error = norm_1(term);
    ++k;
  }
  return B;
}

// Project to O(N) (Orthogonal Group) using a Newton-type algorithm.
// See Higham's Functions of Matrices p210 [2008]
// \param A tensor (often a deformation-gradient-like tensor)
// \return \f$ R = \argmin_Q \|A - Q\|\f$
// This algorithm projects a given tensor in GL(N) to O(N).
// The rotation/reflection obtained through this projection is
// the orthogonal component of the real polar decomposition
template <typename T>
HPC_HOST_DEVICE constexpr auto
polar_rotation(matrix3x3<T> const& A)
{
  auto const dim = 3;
  auto scale = true;
  auto const tol_scale = 0.01;
  auto const tol_conv = std::sqrt(dim) * machine_epsilon<T>();
  auto X = A;
  auto gamma = 2.0;
  auto const max_iter = 128;
  auto num_iter = 0;
  while (num_iter < max_iter) {
    auto const Y = inverse_full_pivot(X);
    auto mu = 1.0;
    if (scale == true) {
      mu = (norm_1(Y) * norm_infinity(Y)) / (norm_1(X) * norm_infinity(X));
      mu = std::sqrt(std::sqrt(mu));
    }
    auto const Z = 0.5 * (mu * X + transpose(Y) / mu);
    auto const D = Z - X;
    auto const delta = norm(D) / norm(Z);
    if (scale == true && delta < tol_scale) {
      scale = false;
    }
    auto const end_iter = norm(D) <= std::sqrt(tol_conv) ||
        (delta > 0.5 * gamma && scale == false);
    X = Z;
    gamma = delta;
    if (end_iter == true) {
      break;
    }
    num_iter++;
  }
  return X;
}

template <typename T>
HPC_HOST_DEVICE constexpr auto
symm(matrix3x3<T> const A)
{
  return 0.5 * (A + transpose(A));
}

template <typename T>
HPC_HOST_DEVICE constexpr auto
skew(matrix3x3<T> const A)
{
  return 0.5 * (A - transpose(A));
}

template <typename T>
HPC_HOST_DEVICE constexpr auto
check(vector3<T> const w)
{
  return matrix3x3<T>(0.0, -w(2), w(1), w(2), 0.0, -w(0), -w(1), w(0), 0.0);
}

template <typename T>
HPC_HOST_DEVICE constexpr auto
uncheck(matrix3x3<T> const A)
{
  auto const W = skew(A);
  return vector3<T>(W(2, 1), W(0, 2), W(1, 0));
}

template <typename T>
HPC_HOST constexpr auto
polar_left(matrix3x3<T> const A)
{
  auto const R = polar_rotation(A);
  auto const V = symm(A * transpose(R));
  return std::make_pair(V, R);
}

template <typename T>
HPC_HOST constexpr auto
polar_right(matrix3x3<T> const A)
{
  auto const R = polar_rotation(A);
  auto const U = symm(transpose(R) * A);
  return std::make_pair(R, U);
}

template <class T>
HPC_HOST_DEVICE constexpr T
trace(matrix3x3<T> x) noexcept {
  return x(0, 0) + x(1, 1) + x(2, 2);
}

template <class T>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr matrix3x3<T>
isotropic_part(matrix3x3<T> const x) noexcept {
  return ((1.0 / 3.0) * trace(x)) * matrix3x3<T>::identity();
}

template <class T>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr matrix3x3<T>
vol(matrix3x3<T> const A) noexcept {
  return isotropic_part(A);
}

template <class T>
HPC_HOST_DEVICE constexpr matrix3x3<T>
deviatoric_part(matrix3x3<T> x) noexcept {
  auto x_dev = matrix3x3<T>(x);
  auto const a = (1.0 / 3.0) * trace(x);
  x_dev(0,0) -= a;
  x_dev(1,1) -= a;
  x_dev(2,2) -= a;
  return x_dev;
}

template <class T>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr matrix3x3<T>
dev(matrix3x3<T> const A) noexcept {
  return deviatoric_part(A);
}

template <class T>
class array_traits<matrix3x3<T>> {
  public:
  using value_type = T;
  using size_type = decltype(axis_index() * axis_index());
  HPC_HOST_DEVICE static constexpr size_type size() noexcept { return 9; }
  template <class Iterator>
  HPC_HOST_DEVICE static matrix3x3<T> load(Iterator it) noexcept {
    return matrix3x3<T>(
        it[0],
        it[1],
        it[2],
        it[3],
        it[4],
        it[5],
        it[6],
        it[7],
        it[8]);
  }
  template <class Iterator>
  HPC_HOST_DEVICE static void store(Iterator it, matrix3x3<T> const& value) noexcept {
    it[0] = value(0, 0);
    it[1] = value(0, 1);
    it[2] = value(0, 2);
    it[3] = value(1, 0);
    it[4] = value(1, 1);
    it[5] = value(1, 2);
    it[6] = value(2, 0);
    it[7] = value(2, 1);
    it[8] = value(2, 2);
  }
};

// Baker-Campbell-Hausdorff formula, up to 8 terms.
//
// The Baker–Campbell–Hausdorff formula is the solution to the equation
//
// z = log[exp(x) exp(y)]
//
// for possibly noncommutative "x" and "y" in the Lie algebra of a Lie
// group. This formula tightly links Lie groups to Lie algebras by
// expressing the logarithm of the product of two Lie group elements as
// a Lie algebra element using only Lie algebraic operations. The
// solution on this form, whenever defined, means that multiplication
// in the group can be expressed entirely in Lie algebraic terms. The
// solution on commutative forms is obtained by substituting the power
// series for exp and log in the equation and rearranging. The point
// is to express the solution in Lie algebraic terms.
//
// The coefficients on the series were computed by using the
// Mathematica implementation of Goldberg's algorithm given in:
// Computing the Baker-Campbell-Hausdorff series and the Zassenhaus
// product, Weyrauch, Michael and Scholz, Daniel, COMPUTER PHYSICS
// COMMUNICATIONS, 2009, 180:9,1558-1565.
//
template <typename T>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr matrix3x3<T>
bch(matrix3x3<T> const& x, matrix3x3<T> const& y) noexcept {
  auto const z1 = x+y;

  auto const z2 = 0.5*(x*y - y*x);

  auto const z3 = x*x*y/12 - x*y*x/6 + x*y*y/12 + y*x*x/12 - y*x*y/6 + y*y*x/12;

  auto const z4 = x*x*y*y/24 - x*y*x*y/12 + y*x*y*x/12 - y*y*x*x/24;

  auto const z5 = -x*x*x*x*y/720 + x*x*x*y*x/180 + x*x*x*y*y/180 - x*x*y*x*x/120 -
  x*x*y*x*y/120 - x*x*y*y*x/120 + x*x*y*y*y/180 + x*y*x*x*x/180 -
  x*y*x*x*y/120 + x*y*x*y*x/30 - x*y*x*y*y/120 - x*y*y*x*x/120 - x*y*y*x*y/120
  + x*y*y*y*x/180 - x*y*y*y*y/720 - y*x*x*x*x/720 + y*x*x*x*y/180 -
  y*x*x*y*x/120 - y*x*x*y*y/120 - y*x*y*x*x/120 + y*x*y*x*y/30 - y*x*y*y*x/120
  + y*x*y*y*y/180 + y*y*x*x*x/180 - y*y*x*x*y/120 - y*y*x*y*x/120 -
  y*y*x*y*y/120 + y*y*y*x*x/180 + y*y*y*x*y/180 - y*y*y*y*x/720;

  auto const z6 = -x*x*x*x*y*y/1440 + x*x*x*y*x*y/360 + x*x*x*y*y*y/360 - x*x*y*x*x*y/240
  - x*x*y*x*y*y/240 - x*x*y*y*x*y/240 - x*x*y*y* y*y/1440 + x*y*x*x*x*y/360 -
  x*y*x*x*y*y/240 + x*y*x*y*x*y/60 + x*y*x*y*y*y/360 - x*y*y*x*x*y/240 -
  x*y*y*x*y*y/240 + x*y*y*y*x*y/360 - y*x*x*x*y*x/360 + y*x*x*y*x*x/240 +
  y*x*x*y*y*x/240 - y*x*y*x*x*x/360 - y*x*y*x*y*x/60 + y*x*y*y*x*x/240 -
  y*x*y*y*y*x/360 + y*y*x*x*x*x/1440 + y*y*x*x*y*x/240 + y*y*x*y*x*x/240 +
  y*y*x*y*y*x/240 - y*y*y*x*x*x/360 - y*y*y*x*y*x/360 + y*y*y*y*x*x/1440;

  auto const z7 = x*x*x*x*x*x*y/30240 - x*x*x*x*x*y*x/5040 - x*x*x*x*x*y*y/5040 +
  x*x*x*x*y*x*x/2016 + x*x*x*x*y*x*y/2016 + x*x*x*x*y*y*x/2016 +
  x*x*x*x*y*y*y/3780 - x*x*x*y*x*x*x/1512 - x*x*x*y*x*x*y/5040 -
  x*x*x*y*x*y*x/630 - x*x*x*y*x*y*y/5040 - x*x*x*y*y*x*x/5040 -
  x*x*x*y*y*x*y/5040 - x*x*x*y*y*y*x/1512 + x*x*x*y*y*y*y/3780 +
  x*x*y*x*x*x*x/2016 - x*x*y*x*x*x*y/5040 + x*x*y*x*x*y*x/840 -
  x*x*y*x*x*y*y/1120 + x*x*y*x*y*x*x/840 + x*x*y*x*y*x*y/840 +
  x*x*y*x*y*y*x/840 - x*x*y*x*y*y*y/5040 - x*x*y*y*x*x*x/5040 -
  x*x*y*y*x*x*y/1120 + x*x*y*y*x*y*x/840 - x*x*y*y*x*y*y/1120 -
  x*x*y*y*y*x*x/5040 - x*x*y*y*y*x*y/5040 + x*x*y*y*y*y*x/2016 -
  x*x*y*y*y*y*y/5040 - x*y*x*x*x*x*x/5040 + x*y*x*x*x*x*y/2016 -
  x*y*x*x*x*y*x/630 - x*y*x*x*x*y*y/5040 + x*y*x*x*y*x*x/840 +
  x*y*x*x*y*x*y/840 + x*y*x*x*y*y*x/840 - x*y*x*x*y*y*y/5040 -
  x*y*x*y*x*x*x/630 + x*y*x*y*x*x*y/840 - x*y*x*y*x*y*x/140 +
  x*y*x*y*x*y*y/840 + x*y*x*y*y*x*x/840 + x*y*x*y*y*x*y/840 -
  x*y*x*y*y*y*x/630 + x*y*x*y*y*y*y/2016 + x*y*y*x*x*x*x/2016 -
  x*y*y*x*x*x*y/5040 + x*y*y*x*x*y*x/840 - x*y*y*x*x*y*y/1120 +
  x*y*y*x*y*x*x/840 + x*y*y*x*y*x*y/840 + x*y*y*x*y*y*x/840 -
  x*y*y*x*y*y*y/5040 - x*y*y*y*x*x*x/1512 - x*y*y*y*x*x*y/5040 -
  x*y*y*y*x*y*x/630 - x*y*y*y*x*y*y/5040 + x*y*y*y*y*x*x/2016 +
  x*y*y*y*y*x*y/2016 - x*y*y*y*y*y*x/5040 + x*y*y*y*y*y*y/30240 +
  y*x*x*x*x*x*x/30240 - y*x*x*x*x*x*y/5040 + y*x*x*x*x*y*x/2016 +
  y*x*x*x*x*y*y/2016 - y*x*x*x*y*x*x/5040 - y*x*x*x*y*x*y/630 -
  y*x*x*x*y*y*x/5040 - y*x*x*x*y*y*y/1512 - y*x*x*y*x*x*x/5040 +
  y*x*x*y*x*x*y/840 + y*x*x*y*x*y*x/840 + y*x*x*y*x*y*y/840 -
  y*x*x*y*y*x*x/1120 + y*x*x*y*y*x*y/840 - y*x*x*y*y*y*x/5040 +
  y*x*x*y*y*y*y/2016 + y*x*y*x*x*x*x/2016 - y*x*y*x*x*x*y/630 +
  y*x*y*x*x*y*x/840 + y*x*y*x*x*y*y/840 + y*x*y*x*y*x*x/840 -
  y*x*y*x*y*x*y/140 + y*x*y*x*y*y*x/840 - y*x*y*x*y*y*y/630 -
  y*x*y*y*x*x*x/5040 + y*x*y*y*x*x*y/840 + y*x*y*y*x*y*x/840 +
  y*x*y*y*x*y*y/840 - y*x*y*y*y*x*x/5040 - y*x*y*y*y*x*y/630 +
  y*x*y*y*y*y*x/2016 - y*x*y*y*y*y*y/5040 - y*y*x*x*x*x*x/5040 +
  y*y*x*x*x*x*y/2016 - y*y*x*x*x*y*x/5040 - y*y*x*x*x*y*y/5040 -
  y*y*x*x*y*x*x/1120 + y*y*x*x*y*x*y/840 - y*y*x*x*y*y*x/1120 -
  y*y*x*x*y*y*y/5040 - y*y*x*y*x*x*x/5040 + y*y*x*y*x*x*y/840 +
  y*y*x*y*x*y*x/840 + y*y*x*y*x*y*y/840 - y*y*x*y*y*x*x/1120 +
  y*y*x*y*y*x*y/840 - y*y*x*y*y*y*x/5040 + y*y*x*y*y*y*y/2016 +
  y*y*y*x*x*x*x/3780 - y*y*y*x*x*x*y/1512 - y*y*y*x*x*y*x/5040 -
  y*y*y*x*x*y*y/5040 - y*y*y*x*y*x*x/5040 - y*y*y*x*y*x*y/630 -
  y*y*y*x*y*y*x/5040 - y*y*y*x*y*y*y/1512 + y*y*y*y*x*x*x/3780 +
  y*y*y*y*x*x*y/2016 + y*y*y*y*x*y*x/2016 + y*y*y*y*x*y*y/2016 -
  y*y*y*y*y*x*x/5040 - y*y*y*y*y*x*y/5040 + y*y*y*y*y*y*x/30240;

  auto const z8 = x*x*x*x*x*x*y*y/60480 - x*x*x*x*x*y*x*y/10080 - x*x*x*x*x*y*y*y/10080 +
  x*x*x*x*y*x*x*y/4032 + x*x*x*x*y*x*y*y/4032 + x*x*x*x*y*y*x*y/4032 +
  23*x*x*x*x*y*y*y*y/120960 - x*x*x*y*x*x*x*y/3024 - x*x*x*y*x*x*y*y/10080 -
  x*x*x*y*x*y*x*y/1260 - x*x*x*y*x*y*y*y/3024 - x*x*x*y*y*x*x*y/10080 -
  x*x*x*y*y*x*y*y/10080 - x*x*x*y*y*y*x*y/3024 - x*x*x*y*y*y*y*y/10080 +
  x*x*y*x*x*x*x*y/4032 - x*x*y*x*x*x*y*y/10080 + x*x*y*x*x*y*x*y/1680 -
  x*x*y*x*x*y*y*y/10080 + x*x*y*x*y*x*x*y/1680 + x*x*y*x*y*x*y*y/1680 +
  x*x*y*x*y*y*x*y/1680 + x*x*y*x*y*y*y*y/4032 - x*x*y*y*x*x*x*y/10080 -
  x*x*y*y*x*x*y*y/2240 + x*x*y*y*x*y*x*y/1680 - x*x*y*y*x*y*y*y/10080 -
  x*x*y*y*y*x*x*y/10080 - x*x*y*y*y*x*y*y/10080 + x*x*y*y*y*y*x*y/4032 +
  x*x*y*y*y*y*y*y/60480 - x*y*x*x*x*x*x*y/10080 + x*y*x*x*x*x*y*y/4032 -
  x*y*x*x*x*y*x*y/1260 - x*y*x*x*x*y*y*y/3024 + x*y*x*x*y*x*x*y/1680 +
  x*y*x*x*y*x*y*y/1680 + x*y*x*x*y*y*x*y/1680 + x*y*x*x*y*y*y*y/4032 -
  x*y*x*y*x*x*x*y/1260 + x*y*x*y*x*x*y*y/1680 - x*y*x*y*x*y*x*y/280 -
  x*y*x*y*x*y*y*y/1260 + x*y*x*y*y*x*x*y/1680 + x*y*x*y*y*x*y*y/1680 -
  x*y*x*y*y*y*x*y/1260 - x*y*x*y*y*y*y*y/10080 + x*y*y*x*x*x*x*y/4032 -
  x*y*y*x*x*x*y*y/10080 + x*y*y*x*x*y*x*y/1680 - x*y*y*x*x*y*y*y/10080 +
  x*y*y*x*y*x*x*y/1680 + x*y*y*x*y*x*y*y/1680 + x*y*y*x*y*y*x*y/1680 +
  x*y*y*x*y*y*y*y/4032 - x*y*y*y*x*x*x*y/3024 - x*y*y*y*x*x*y*y/10080 -
  x*y*y*y*x*y*x*y/1260 - x*y*y*y*x*y*y*y/3024 + x*y*y*y*y*x*x*y/4032 +
  x*y*y*y*y*x*y*y/4032 - x*y*y*y*y*y*x*y/10080 + y*x*x*x*x*x*y*x/10080 -
  y*x*x*x*x*y*x*x/4032 - y*x*x*x*x*y*y*x/4032 + y*x*x*x*y*x*x*x/3024 +
  y*x*x*x*y*x*y*x/1260 + y*x*x*x*y*y*x*x/10080 + y*x*x*x*y*y*y*x/3024 -
  y*x*x*y*x*x*x*x/4032 - y*x*x*y*x*x*y*x/1680 - y*x*x*y*x*y*x*x/1680 -
  y*x*x*y*x*y*y*x/1680 + y*x*x*y*y*x*x*x/10080 - y*x*x*y*y*x*y*x/1680 +
  y*x*x*y*y*y*x*x/10080 - y*x*x*y*y*y*y*x/4032 + y*x*y*x*x*x*x*x/10080 +
  y*x*y*x*x*x*y*x/1260 - y*x*y*x*x*y*x*x/1680 - y*x*y*x*x*y*y*x/1680 +
  y*x*y*x*y*x*x*x/1260 + y*x*y*x*y*x*y*x/280 - y*x*y*x*y*y*x*x/1680 +
  y*x*y*x*y*y*y*x/1260 - y*x*y*y*x*x*x*x/4032 - y*x*y*y*x*x*y*x/1680 -
  y*x*y*y*x*y*x*x/1680 - y*x*y*y*x*y*y*x/1680 + y*x*y*y*y*x*x*x/3024 +
  y*x*y*y*y*x*y*x/1260 - y*x*y*y*y*y*x*x/4032 + y*x*y*y*y*y*y*x/10080 -
  y*y*x*x*x*x*x*x/60480 - y*y*x*x*x*x*y*x/4032 + y*y*x*x*x*y*x*x/10080 +
  y*y*x*x*x*y*y*x/10080 + y*y*x*x*y*x*x*x/10080 - y*y*x*x*y*x*y*x/1680 +
  y*y*x*x*y*y*x*x/2240 + y*y*x*x*y*y*y*x/10080 - y*y*x*y*x*x*x*x/4032 -
  y*y*x*y*x*x*y*x/1680 - y*y*x*y*x*y*x*x/1680 - y*y*x*y*x*y*y*x/1680 +
  y*y*x*y*y*x*x*x/10080 - y*y*x*y*y*x*y*x/1680 + y*y*x*y*y*y*x*x/10080 -
  y*y*x*y*y*y*y*x/4032 + y*y*y*x*x*x*x*x/10080 + y*y*y*x*x*x*y*x/3024 +
  y*y*y*x*x*y*x*x/10080 + y*y*y*x*x*y*y*x/10080 + y*y*y*x*y*x*x*x/3024 +
  y*y*y*x*y*x*y*x/1260 + y*y*y*x*y*y*x*x/10080 + y*y*y*x*y*y*y*x/3024 -
  23*y*y*y*y*x*x*x*x/120960 - y*y*y*y*x*x*y*x/4032 - y*y*y*y*x*y*x*x/4032 -
  y*y*y*y*x*y*y*x/4032 + y*y*y*y*y*x*x*x/10080 + y*y*y*y*y*x*y*x/10080 -
  y*y*y*y*y*y*x*x/60480;

  auto const z=z1+z2+z3+z4+z5+z6+z7+z8;

  return z;
}

}
