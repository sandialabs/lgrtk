#pragma once

#include <lgr_matrix3x3.hpp>

#include <iostream>

namespace lgr {

template <typename Scalar>
struct symmetric3x3 {
public:
  using scalar_type = Scalar;
  enum slot_type {
    XX = 0,
    YY = 1,
    ZZ = 2,
    XY = 3,
    YZ = 4,
    XZ = 5,
  };
private:
  scalar_type raw[6];
public:
  explicit constexpr inline symmetric3x3(
      Scalar const a, Scalar const b, Scalar const c,
      Scalar const d, Scalar const e, Scalar const f,
      Scalar const g, Scalar const h, Scalar const i) noexcept
  {
    raw[XX] = a;
    raw[YY] = e;
    raw[ZZ] = i;
    raw[XY] = (b + d) * 0.5;
    raw[YZ] = (f + h) * 0.5;
    raw[XZ] = (c + g) * 0.5;
  }
  explicit constexpr inline symmetric3x3(
      Scalar const xx, Scalar const yy, Scalar const zz,
      Scalar const xy, Scalar const yz, Scalar const xz) noexcept
  {
    raw[XX] = xx;
    raw[YY] = yy;
    raw[ZZ] = zz;
    raw[XY] = xy;
    raw[YZ] = yz;
    raw[XZ] = xz;
  }
  explicit constexpr inline symmetric3x3(matrix3x3<Scalar> const t):
    symmetric3x3(
        t(0, 0), t(0, 1), t(0, 2),
        t(1, 0), t(1, 1), t(1, 2),
        t(2, 0), t(2, 1), t(2, 2))
  {
  }
  inline symmetric3x3() = default;
  constexpr inline matrix3x3<Scalar> full() const noexcept {
    return matrix3x3<Scalar>(
        raw[XX], raw[XY], raw[XZ],
        raw[XY], raw[YY], raw[YZ],
        raw[XZ], raw[YZ], raw[ZZ]);
  }
  constexpr inline Scalar operator()(slot_type const slot) const noexcept {
    return raw[slot];
  }
  static constexpr symmetric3x3<Scalar> zero() noexcept {
    return symmetric3x3<Scalar>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
  }
  inline Scalar operator()(int const i, int const j) const noexcept {
    if (i == 0 && j == 0) return raw[XX];
    if (i == 0 && j == 1) return raw[XY];
    if (i == 0 && j == 2) return raw[XZ];
    if (i == 1 && j == 0) return raw[XY];
    if (i == 1 && j == 1) return raw[YY];
    if (i == 1 && j == 2) return raw[YZ];
    if (i == 2 && j == 0) return raw[XZ];
    if (i == 2 && j == 1) return raw[YZ];
    if (i == 2 && j == 2) return raw[ZZ];
    return -1.0;
  }
};

template <typename Scalar>
constexpr inline symmetric3x3<Scalar>
operator+(symmetric3x3<Scalar> const left,
    symmetric3x3<Scalar> const right) noexcept {
  using st = symmetric3x3<Scalar>;
  return st(
      left(st::XX) + right(st::XX),
      left(st::YY) + right(st::YY),
      left(st::ZZ) + right(st::ZZ),
      left(st::XY) + right(st::XY),
      left(st::YZ) + right(st::YZ),
      left(st::XZ) + right(st::XZ));
}

template <typename Scalar>
constexpr inline symmetric3x3<Scalar>
operator+(symmetric3x3<Scalar> const left,
    Scalar const right) noexcept {
  using st = symmetric3x3<Scalar>;
  return st(
      left(st::XX) + right,
      left(st::YY) + right,
      left(st::ZZ) + right,
      left(st::XY),
      left(st::YZ),
      left(st::XZ));
}

template <typename Scalar>
constexpr inline symmetric3x3<Scalar>
operator+(Scalar const left,
    symmetric3x3<Scalar> const right) noexcept {
  return right + left;
}

template <typename Scalar>
constexpr inline symmetric3x3<Scalar>
operator-(symmetric3x3<Scalar> const left,
    symmetric3x3<Scalar> const right) noexcept {
  using st = symmetric3x3<Scalar>;
  return st(
      left(st::XX) - right(st::XX),
      left(st::YY) - right(st::YY),
      left(st::ZZ) - right(st::ZZ),
      left(st::XY) - right(st::XY),
      left(st::YZ) - right(st::YZ),
      left(st::XZ) - right(st::XZ));
}

template <typename Scalar>
constexpr inline symmetric3x3<Scalar>
operator-(symmetric3x3<Scalar> const left,
    Scalar const right) noexcept {
  using st = symmetric3x3<Scalar>;
  return st(
      left(st::XX) - right,
      left(st::YY) - right,
      left(st::ZZ) - right,
      left(st::XY),
      left(st::YZ),
      left(st::XZ));
}

template <typename Scalar>
constexpr inline matrix3x3<Scalar>
operator*(symmetric3x3<Scalar> const left,
    symmetric3x3<Scalar> const right) noexcept {
  return left.full() * right.full();
}

template <typename Scalar>
inline Scalar
inner_product(symmetric3x3<Scalar> const left,
    symmetric3x3<Scalar> const right) noexcept {
  using st = symmetric3x3<Scalar>;
  Scalar const diagonal_inner_product =
    (left(st::XX) * right(st::XX)) +
    (left(st::YY) * right(st::YY)) +
    (left(st::ZZ) * right(st::ZZ));
  Scalar const triangular_inner_product =
    (left(st::XY) * right(st::XY)) +
    (left(st::YZ) * right(st::YZ)) +
    (left(st::XZ) * right(st::XZ));
//std::cout << "diagonal_inner_product " << diagonal_inner_product << '\n';
//std::cout << "triangular_inner_product " << triangular_inner_product << '\n';
  return diagonal_inner_product + (2.0 * triangular_inner_product);
}

template <typename Scalar>
constexpr inline matrix3x3<Scalar>
operator*(matrix3x3<Scalar> const left,
    symmetric3x3<Scalar> const right) noexcept {
  return left * right.full();
}

template <typename Scalar>
constexpr inline matrix3x3<Scalar>
operator*(symmetric3x3<Scalar> const left,
    matrix3x3<Scalar> const right) noexcept {
  return right * left;
}

template <typename Scalar>
constexpr inline symmetric3x3<Scalar>
self_times_transpose(matrix3x3<Scalar> const in) noexcept {
  return symmetric3x3<Scalar>(in * transpose(in));
}

template <typename Scalar>
constexpr inline symmetric3x3<Scalar>
transpose_times_self(matrix3x3<Scalar> const in) noexcept {
  return transpose(in) * in;
}

template <typename Scalar>
constexpr inline vector3<Scalar>
operator*(symmetric3x3<Scalar> const left,
    vector3<Scalar> const right) noexcept {
  using out_t = vector3<Scalar>;
  using st = decltype(left);
  return out_t(
      left(st::XX) * right(0) + left(st::XY) * right(1) + left(st::XZ) * right(2),
      left(st::XY) * right(0) + left(st::YY) * right(1) + left(st::YZ) * right(2),
      left(st::XZ) * right(0) + left(st::YZ) * right(1) + left(st::ZZ) * right(2));
}

template <typename Scalar>
constexpr inline vector3<Scalar>
operator*(vector3<Scalar> const left,
    symmetric3x3<Scalar> const right) noexcept {
  return right * left;
}

template <typename Scalar>
constexpr inline symmetric3x3<Scalar>
operator*(symmetric3x3<Scalar> const left,
    Scalar const right) noexcept {
  using st = symmetric3x3<Scalar>;
  return st(
      left(st::XX) * right,
      left(st::YY) * right,
      left(st::ZZ) * right,
      left(st::XY) * right,
      left(st::YZ) * right,
      left(st::XZ) * right);
}

template <typename Scalar>
constexpr inline symmetric3x3<Scalar>
operator*(Scalar const left,
    symmetric3x3<Scalar> const right) noexcept {
  return right * left;
}

template <typename Scalar>
constexpr inline symmetric3x3<Scalar>
operator/(symmetric3x3<Scalar> const left,
    Scalar const right) noexcept {
  using st = symmetric3x3<Scalar>;
  Scalar const factor = 1.0 / right;
  return st(
      left(st::XX) * factor,
      left(st::YY) * factor,
      left(st::ZZ) * factor,
      left(st::XY) * factor,
      left(st::YZ) * factor,
      left(st::XZ) * factor);
}

template <typename Scalar>
constexpr inline Scalar
trace(symmetric3x3<Scalar> const x) noexcept {
  using out_t = Scalar;
  using st = symmetric3x3<Scalar>;
  return out_t(x(st::XX) + x(st::YY) + x(st::ZZ));
}

template <typename Scalar>
constexpr inline symmetric3x3<Scalar>
deviator(symmetric3x3<Scalar> const x) noexcept {
  return x - ((1.0 / 3.0) * trace(x));
}

}
