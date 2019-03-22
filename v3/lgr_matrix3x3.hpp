#pragma once

#include <lgr_vector3.hpp>

namespace lgr {

template <typename Scalar>
class matrix3x3 {
public:
  using scalar_type = Scalar;
private:
  scalar_type raw[3][3];
public:
  constexpr inline matrix3x3(
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
  inline matrix3x3() noexcept = default;
  static constexpr inline matrix3x3 identity() noexcept {
    return matrix3x3(
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0);
  }
  static constexpr inline matrix3x3 zero() noexcept {
    return matrix3x3(
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0);
  }
  constexpr inline Scalar operator()(int const i, int const j) const noexcept {
    return raw[i][j];
  }
};

template <typename Scalar>
constexpr inline matrix3x3<Scalar>
operator+(matrix3x3<Scalar> const left,
    matrix3x3<Scalar> const right) noexcept {
  return matrix3x3<Scalar>(
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

template <typename Scalar>
constexpr inline matrix3x3<Scalar>
operator+(matrix3x3<Scalar> const left,
    Scalar const right) noexcept {
  return matrix3x3<Scalar>(
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

template <typename Scalar>
constexpr inline matrix3x3<Scalar>
operator+(Scalar const left,
    matrix3x3<Scalar> const right) noexcept {
  return right + left;
}

template <typename Scalar>
constexpr inline matrix3x3<Scalar>
operator-(matrix3x3<Scalar> const left,
    matrix3x3<Scalar> const right) noexcept {
  return matrix3x3<Scalar>(
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

template <typename Scalar>
constexpr inline matrix3x3<Scalar>
operator-(matrix3x3<Scalar> const left,
    Scalar const right) noexcept {
  return matrix3x3<Scalar>(
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

template <typename Scalar>
constexpr inline matrix3x3<Scalar>
operator*(matrix3x3<Scalar> const left,
    matrix3x3<Scalar> const right) noexcept {
  return matrix3x3<Scalar>(
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

template <typename Scalar>
constexpr inline vector3<Scalar>
operator*(matrix3x3<Scalar> const left,
    vector3<Scalar> const right) noexcept {
  return vector3<Scalar>(
      left(0, 0) * right(0) + left(0, 1) * right(1) + left(0, 2) * right(2),
      left(1, 0) * right(0) + left(1, 1) * right(1) + left(1, 2) * right(2),
      left(2, 0) * right(0) + left(2, 1) * right(1) + left(2, 2) * right(2));
}

template <typename Scalar>
constexpr inline vector3<Scalar>
operator*(vector3<Scalar> const left,
    matrix3x3<Scalar> const right) noexcept {
  return vector3<Scalar>(
      left(0) * right(0, 0) + left(1) * right(1, 0) + left(2) * right(2, 0),
      left(0) * right(0, 1) + left(1) * right(1, 1) + left(2) * right(2, 1),
      left(0) * right(0, 2) + left(1) * right(1, 2) + left(2) * right(2, 2));
}

template <typename Scalar>
constexpr inline matrix3x3<Scalar>
operator*(matrix3x3<Scalar> const left,
    Scalar const right) noexcept {
  return matrix3x3<Scalar>(
      left(0, 0) * right, left(0, 1) * right, left(0, 2) * right,
      left(1, 0) * right, left(1, 1) * right, left(1, 2) * right,
      left(2, 0) * right, left(2, 1) * right, left(2, 2) * right);
}

template <typename Scalar>
constexpr inline matrix3x3<Scalar>
operator*(Scalar const left,
    matrix3x3<Scalar> const right) noexcept {
  return right * left;
}

template <typename Scalar>
constexpr inline matrix3x3<Scalar>
operator/(matrix3x3<Scalar> const left,
    Scalar const right) noexcept {
  Scalar const factor = 1.0 / right;
  return matrix3x3<Scalar>(
      left(0, 0) * factor, left(0, 1) * factor, left(0, 2) * factor,
      left(1, 0) * factor, left(1, 1) * factor, left(1, 2) * factor,
      left(2, 0) * factor, left(2, 1) * factor, left(2, 2) * factor);
}

template <typename Scalar>
constexpr inline matrix3x3<Scalar>
transpose(matrix3x3<Scalar> const in) noexcept {
  return matrix3x3<Scalar>(
      in(0, 0),
      in(1, 0),
      in(2, 0),
      in(0, 1),
      in(1, 1),
      in(2, 1),
      in(0, 2),
      in(1, 2),
      in(2, 2));
}

template <typename Scalar>
constexpr inline matrix3x3<Scalar>
outer_product(vector3<Scalar> const left,
    vector3<Scalar> const right) noexcept {
  return matrix3x3<Scalar>(
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
constexpr inline Scalar
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

template <typename Scalar>
constexpr inline matrix3x3<Scalar>
inverse(matrix3x3<Scalar> const x) {
  Scalar const a = x(0, 0);
  Scalar const b = x(0, 1);
  Scalar const c = x(0, 2);
  Scalar const d = x(1, 0);
  Scalar const e = x(1, 1);
  Scalar const f = x(1, 2);
  Scalar const g = x(2, 0);
  Scalar const h = x(2, 1);
  Scalar const i = x(2, 2);
  Scalar const A = (e * i - f * h);
  Scalar const D = -(b * i - c * h);
  Scalar const G = (b * f - c * e);
  Scalar const B = -(d * i - f * g);
  Scalar const E = (a * i - c * g);
  Scalar const H = -(a * f - c * d);
  Scalar const C = (d * h - e * g);
  Scalar const F = -(a * h - b * g);
  Scalar const I = (a * e - b * d);
  using out_t = matrix3x3<Scalar>;
  auto const inverse_times_det = out_t(A, D, G, B, E, H, C, F, I);
  return inverse_times_det / determinant(x);
}

template <typename Scalar>
constexpr inline Scalar
trace(matrix3x3<Scalar> const x) noexcept {
  using out_t = Scalar;
  return out_t(x(0, 0) + x(1, 1) + x(2, 2));
}

template <typename Scalar>
constexpr inline matrix3x3<Scalar>
deviator(matrix3x3<Scalar> const x) noexcept {
  Scalar const factor = ((1.0 / 3.0) * trace(x));
  return matrix3x3<Scalar>(
      x(0, 0) - factor, x(0, 1), x(0, 2),
      x(1, 0), x(1, 1) - factor, x(1, 2),
      x(2, 0), x(2, 1), x(2, 2) - factor);
}

}
