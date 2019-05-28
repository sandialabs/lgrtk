#pragma once

#include <hpc_vector3.hpp>
#include <hpc_array_traits.hpp>

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
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE matrix3x3() noexcept = default;
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

template <class T>
HPC_HOST_DEVICE constexpr matrix3x3<T>
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
inverse(matrix3x3<T> const x) {
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
  using denom_t = matrix3x3<std::remove_const_t<decltype(A)>>;
  auto const denom = denom_t(A, D, G, B, E, H, C, F, I);
  return denom / determinant(x);
}

template <class T>
HPC_HOST_DEVICE constexpr T
trace(matrix3x3<T> x) noexcept {
  return x(0, 0) + x(1, 1) + x(2, 2);
}

template <class T>
HPC_HOST_DEVICE constexpr matrix3x3<T>
deviator(matrix3x3<T> x) noexcept {
  return x - ((1.0 / 3.0) * trace(x));
}

template <class T>
class array_traits<matrix3x3<T>> {
  public:
  using value_type = T;
  HPC_HOST_DEVICE static constexpr std::ptrdiff_t size() noexcept { return 9; }
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

}
