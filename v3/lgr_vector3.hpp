#pragma once

#include <cmath>

namespace lgr {

template <typename Scalar>
class vector3 {
public:
  using scalar_type = Scalar;
private:
  scalar_type raw[3];
public:
  explicit constexpr inline vector3(Scalar const x, Scalar const y, Scalar const z) noexcept
    :raw{x, y, z}
  {
  }
  inline vector3() noexcept = default;
  constexpr inline scalar_type operator()(int const i) const noexcept { return raw[i]; }
  static constexpr inline vector3 zero() noexcept { return vector3(0.0, 0.0, 0.0); }
};

template <typename Scalar>
constexpr inline vector3<Scalar>
operator+(vector3<Scalar> const left,
    vector3<Scalar> const right) noexcept {
  return vector3<Scalar>(
      left(0) + right(0),
      left(1) + right(1),
      left(2) + right(2));
}

template <typename Scalar>
constexpr inline vector3<Scalar>
operator-(vector3<Scalar> const x) noexcept {
  return vector3<Scalar>(-x(0), -x(1), -x(2));
}

template <typename Scalar>
constexpr inline vector3<Scalar>
operator-(vector3<Scalar> const left,
    vector3<Scalar> const right) noexcept {
  return vector3<Scalar>(
      left(0) - right(0),
      left(1) - right(1),
      left(2) - right(2));
}

template <typename Scalar>
constexpr inline Scalar
inner_product(vector3<Scalar> const left,
    vector3<Scalar> const right) noexcept {
  return
      left(0) * right(0) +
      left(1) * right(1) +
      left(2) * right(2);
}

template <typename Scalar>
constexpr inline Scalar
operator*(vector3<Scalar> const left,
    vector3<Scalar> const right) noexcept {
  return inner_product(left, right);
}

template <typename Scalar>
constexpr inline vector3<Scalar>
operator*(vector3<Scalar> const left,
    Scalar const right) noexcept {
  return vector3<Scalar>(
      left(0) * right,
      left(1) * right,
      left(2) * right);
}

template <typename Scalar>
constexpr inline vector3<Scalar>
operator*(Scalar const left,
    vector3<Scalar> const right) noexcept {
  return right * left;
}

template <typename Scalar>
constexpr inline vector3<Scalar>
operator/(vector3<Scalar> const left,
    Scalar const right) noexcept {
  Scalar const factor = 1.0 / right;
  return vector3<Scalar>(
      left(0) * factor,
      left(1) * factor,
      left(2) * factor);
}

template <typename Scalar>
constexpr inline vector3<Scalar>
cross(vector3<Scalar> const left,
    vector3<Scalar> const right) noexcept {
  return vector3<Scalar>(
      left(1) * right(2) - left(2) * right(1),
      left(2) * right(0) - left(0) * right(2),
      left(0) * right(1) - left(1) * right(0));
}

template <typename Scalar>
inline Scalar
norm(vector3<Scalar> const v) noexcept {
  return std::sqrt(v * v);
}

template <typename Scalar>
inline Scalar
normalize(vector3<Scalar> const v) noexcept {
  return v / norm(v);
}

}
