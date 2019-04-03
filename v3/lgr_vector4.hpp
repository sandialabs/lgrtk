#pragma once

namespace lgr {

template <typename Scalar>
class vector4 {
public:
  using scalar_type = Scalar;
private:
  scalar_type raw[4];
public:
  explicit constexpr inline vector4(Scalar const x, Scalar const y, Scalar const z, Scalar const w) noexcept
    :raw{x, y, z, w}
  {
  }
  inline vector4() noexcept = default;
  constexpr inline scalar_type operator()(int const i) const noexcept { return raw[i]; }
  inline scalar_type& operator()(int const i) noexcept { return raw[i]; }
  static constexpr inline vector4 zero() noexcept { return vector4(0.0, 0.0, 0.0, 0.0); }
};

template <typename Scalar>
constexpr inline Scalar
inner_product(vector4<Scalar> const left,
    vector4<Scalar> const right) noexcept {
  return
      left(0) * right(0) +
      left(1) * right(1) +
      left(2) * right(2) +
      left(3) * right(3);
}

template <typename Scalar>
constexpr inline Scalar
operator*(vector4<Scalar> const left,
    vector4<Scalar> const right) noexcept {
  return inner_product(left, right);
}

}
