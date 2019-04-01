#pragma once

namespace lgr {

template <typename Scalar>
class matrix4x4 {
public:
  using scalar_type = Scalar;
private:
  scalar_type raw[4][4];
public:
  constexpr inline matrix4x4(
      Scalar const a, Scalar const b, Scalar const c, Scalar const d,
      Scalar const e, Scalar const f, Scalar const g, Scalar const h,
      Scalar const i, Scalar const j, Scalar const k, Scalar const l) noexcept
  {
    raw[0][0] = a;
    raw[0][1] = b;
    raw[0][2] = c;
    raw[0][3] = d;
    raw[1][0] = e;
    raw[1][1] = f;
    raw[1][2] = g;
    raw[1][3] = h;
    raw[2][0] = i;
    raw[2][1] = j;
    raw[2][2] = k;
    raw[2][3] = l;
  }
  inline matrix4x4() noexcept = default;
  static constexpr inline matrix4x4 identity() noexcept {
    return matrix4x4(
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0);
  }
  static constexpr inline matrix4x4 zero() noexcept {
    return matrix4x4(
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0);
  }
  constexpr inline Scalar operator()(int const i, int const j) const noexcept {
    return raw[i][j];
  }
};

template <typename Scalar>
constexpr inline matrix4x4<Scalar>
operator*(matrix4x4<Scalar> const left,
    Scalar const right) noexcept {
  return matrix4x4<Scalar>(
      left(0, 0) * right, left(0, 1) * right, left(0, 2) * right, left(0, 3) * right,
      left(1, 0) * right, left(1, 1) * right, left(1, 2) * right, left(1, 3) * right,
      left(2, 0) * right, left(2, 1) * right, left(2, 2) * right, left(2, 3) * right,
      left(3, 0) * right, left(3, 1) * right, left(3, 2) * right, left(3, 3) * right);
}

template <typename Scalar>
constexpr inline matrix4x4<Scalar>
operator/(matrix4x4<Scalar> const left,
    Scalar const right) noexcept {
  return left * (1.0 / right);
}

template <class Scalar>
constexpr inline Scalar determinant(matrix4x4 const a) {
  return a(0, 3) * a(1, 2) * a(2, 1) * a(3, 0) -
         a(0, 2) * a(1, 3) * a(2, 1) * a(3, 0) -
         a(0, 3) * a(1, 1) * a(2, 2) * a(3, 0) +
         a(0, 1) * a(1, 3) * a(2, 2) * a(3, 0) +
         a(0, 2) * a(1, 1) * a(2, 3) * a(3, 0) -
         a(0, 1) * a(1, 2) * a(2, 3) * a(3, 0) -
         a(0, 3) * a(1, 2) * a(2, 0) * a(3, 1) +
         a(0, 2) * a(1, 3) * a(2, 0) * a(3, 1) +
         a(0, 3) * a(1, 0) * a(2, 2) * a(3, 1) -
         a(0, 0) * a(1, 3) * a(2, 2) * a(3, 1) -
         a(0, 2) * a(1, 0) * a(2, 3) * a(3, 1) +
         a(0, 0) * a(1, 2) * a(2, 3) * a(3, 1) +
         a(0, 3) * a(1, 1) * a(2, 0) * a(3, 2) -
         a(0, 1) * a(1, 3) * a(2, 0) * a(3, 2) -
         a(0, 3) * a(1, 0) * a(2, 1) * a(3, 2) +
         a(0, 0) * a(1, 3) * a(2, 1) * a(3, 2) +
         a(0, 1) * a(1, 0) * a(2, 3) * a(3, 2) -
         a(0, 0) * a(1, 1) * a(2, 3) * a(3, 2) -
         a(0, 2) * a(1, 1) * a(2, 0) * a(3, 3) +
         a(0, 1) * a(1, 2) * a(2, 0) * a(3, 3) +
         a(0, 2) * a(1, 0) * a(2, 1) * a(3, 3) -
         a(0, 0) * a(1, 2) * a(2, 1) * a(3, 3) -
         a(0, 1) * a(1, 0) * a(2, 2) * a(3, 3) +
         a(0, 0) * a(1, 1) * a(2, 2) * a(3, 3);
}

constexpr inline matrix4x4 inverse(matrix4x4 const a) {
  return matrix4x4(
      (-a(1, 3) * a(2, 2) * a(3, 1) + a(1, 2) * a(2, 3) * a(3, 1) +
          a(1, 3) * a(2, 1) * a(3, 2) - a(1, 1) * a(2, 3) * a(3, 2) -
          a(1, 2) * a(2, 1) * a(3, 3) + a(1, 1) * a(2, 2) * a(3, 3)),
      (a(0, 3) * a(2, 2) * a(3, 1) - a(0, 2) * a(2, 3) * a(3, 1) -
          a(0, 3) * a(2, 1) * a(3, 2) + a(0, 1) * a(2, 3) * a(3, 2) +
          a(0, 2) * a(2, 1) * a(3, 3) - a(0, 1) * a(2, 2) * a(3, 3)),
      (-a(0, 3) * a(1, 2) * a(3, 1) + a(0, 2) * a(1, 3) * a(3, 1) +
          a(0, 3) * a(1, 1) * a(3, 2) - a(0, 1) * a(1, 3) * a(3, 2) -
          a(0, 2) * a(1, 1) * a(3, 3) + a(0, 1) * a(1, 2) * a(3, 3)),
      (a(0, 3) * a(1, 2) * a(2, 1) - a(0, 2) * a(1, 3) * a(2, 1) -
          a(0, 3) * a(1, 1) * a(2, 2) + a(0, 1) * a(1, 3) * a(2, 2) +
          a(0, 2) * a(1, 1) * a(2, 3) - a(0, 1) * a(1, 2) * a(2, 3)),
      (a(1, 3) * a(2, 2) * a(3, 0) - a(1, 2) * a(2, 3) * a(3, 0) -
          a(1, 3) * a(2, 0) * a(3, 2) + a(1, 0) * a(2, 3) * a(3, 2) +
          a(1, 2) * a(2, 0) * a(3, 3) - a(1, 0) * a(2, 2) * a(3, 3)),
      (-a(0, 3) * a(2, 2) * a(3, 0) + a(0, 2) * a(2, 3) * a(3, 0) +
          a(0, 3) * a(2, 0) * a(3, 2) - a(0, 0) * a(2, 3) * a(3, 2) -
          a(0, 2) * a(2, 0) * a(3, 3) + a(0, 0) * a(2, 2) * a(3, 3)),
      (a(0, 3) * a(1, 2) * a(3, 0) - a(0, 2) * a(1, 3) * a(3, 0) -
          a(0, 3) * a(1, 0) * a(3, 2) + a(0, 0) * a(1, 3) * a(3, 2) +
          a(0, 2) * a(1, 0) * a(3, 3) - a(0, 0) * a(1, 2) * a(3, 3)),
      (-a(0, 3) * a(1, 2) * a(2, 0) + a(0, 2) * a(1, 3) * a(2, 0) +
          a(0, 3) * a(1, 0) * a(2, 2) - a(0, 0) * a(1, 3) * a(2, 2) -
          a(0, 2) * a(1, 0) * a(2, 3) + a(0, 0) * a(1, 2) * a(2, 3)),
      (-a(1, 3) * a(2, 1) * a(3, 0) + a(1, 1) * a(2, 3) * a(3, 0) +
          a(1, 3) * a(2, 0) * a(3, 1) - a(1, 0) * a(2, 3) * a(3, 1) -
          a(1, 1) * a(2, 0) * a(3, 3) + a(1, 0) * a(2, 1) * a(3, 3)),
      (a(0, 3) * a(2, 1) * a(3, 0) - a(0, 1) * a(2, 3) * a(3, 0) -
          a(0, 3) * a(2, 0) * a(3, 1) + a(0, 0) * a(2, 3) * a(3, 1) +
          a(0, 1) * a(2, 0) * a(3, 3) - a(0, 0) * a(2, 1) * a(3, 3)),
      (-a(0, 3) * a(1, 1) * a(3, 0) + a(0, 1) * a(1, 3) * a(3, 0) +
          a(0, 3) * a(1, 0) * a(3, 1) - a(0, 0) * a(1, 3) * a(3, 1) -
          a(0, 1) * a(1, 0) * a(3, 3) + a(0, 0) * a(1, 1) * a(3, 3)),
      (a(0, 3) * a(1, 1) * a(2, 0) - a(0, 1) * a(1, 3) * a(2, 0) -
          a(0, 3) * a(1, 0) * a(2, 1) + a(0, 0) * a(1, 3) * a(2, 1) +
          a(0, 1) * a(1, 0) * a(2, 3) - a(0, 0) * a(1, 1) * a(2, 3)),
      (a(1, 2) * a(2, 1) * a(3, 0) - a(1, 1) * a(2, 2) * a(3, 0) -
          a(1, 2) * a(2, 0) * a(3, 1) + a(1, 0) * a(2, 2) * a(3, 1) +
          a(1, 1) * a(2, 0) * a(3, 2) - a(1, 0) * a(2, 1) * a(3, 2)),
      (-a(0, 2) * a(2, 1) * a(3, 0) + a(0, 1) * a(2, 2) * a(3, 0) +
          a(0, 2) * a(2, 0) * a(3, 1) - a(0, 0) * a(2, 2) * a(3, 1) -
          a(0, 1) * a(2, 0) * a(3, 2) + a(0, 0) * a(2, 1) * a(3, 2)),
      (a(0, 2) * a(1, 1) * a(3, 0) - a(0, 1) * a(1, 2) * a(3, 0) -
          a(0, 2) * a(1, 0) * a(3, 1) + a(0, 0) * a(1, 2) * a(3, 1) +
          a(0, 1) * a(1, 0) * a(3, 2) - a(0, 0) * a(1, 1) * a(3, 2)),
      (-a(0, 2) * a(1, 1) * a(2, 0) + a(0, 1) * a(1, 2) * a(2, 0) +
          a(0, 2) * a(1, 0) * a(2, 1) - a(0, 0) * a(1, 2) * a(2, 1) -
          a(0, 1) * a(1, 0) * a(2, 2) + a(0, 0) * a(1, 1) * a(2, 2)))
    / determinant(a);
}

}
