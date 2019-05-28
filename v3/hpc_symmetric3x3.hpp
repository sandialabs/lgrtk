#pragma once

#include <hpc_matrix3x3.hpp>

namespace hpc {

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
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr symmetric3x3(
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
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr symmetric3x3(
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
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE explicit constexpr symmetric3x3(matrix3x3<Scalar> const t):
    symmetric3x3(
        t(0, 0), t(0, 1), t(0, 2),
        t(1, 0), t(1, 1), t(1, 2),
        t(2, 0), t(2, 1), t(2, 2))
  {
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE symmetric3x3() = default;
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr matrix3x3<Scalar> full() const noexcept {
    return matrix3x3<Scalar>(
        raw[XX], raw[XY], raw[XZ],
        raw[XY], raw[YY], raw[YZ],
        raw[XZ], raw[YZ], raw[ZZ]);
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr Scalar operator()(slot_type const slot) const noexcept {
    return raw[slot];
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE static constexpr symmetric3x3<Scalar> zero() noexcept {
    return symmetric3x3<Scalar>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE Scalar operator()(int const i, int const j) const noexcept {
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

template <class T>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr symmetric3x3<T>
operator+(symmetric3x3<T> left, symmetric3x3<T> right) noexcept {
  using st = symmetric3x3<T>;
  return st(
      left(st::XX) + right(st::XX),
      left(st::YY) + right(st::YY),
      left(st::ZZ) + right(st::ZZ),
      left(st::XY) + right(st::XY),
      left(st::YZ) + right(st::YZ),
      left(st::XZ) + right(st::XZ));
}

template <class T>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE symmetric3x3<T>&
operator+=(symmetric3x3<T>& left, symmetric3x3<T> right) noexcept {
  left = left + right;
  return left;
}

template <class L, class R>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr symmetric3x3<L>
operator+(symmetric3x3<L> left, R right) noexcept {
  using st = symmetric3x3<L>;
  return st(
      left(st::XX) + right,
      left(st::YY) + right,
      left(st::ZZ) + right,
      left(st::XY),
      left(st::YZ),
      left(st::XZ));
}

template <class L, class R>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE symmetric3x3<L>&
operator+=(symmetric3x3<L>& left, R right) noexcept {
  left = left + right;
  return left;
}

template <class L, class R>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr auto
operator+(L left, symmetric3x3<R> right) noexcept {
  return right + left;
}

template <class T>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr symmetric3x3<T>
operator-(symmetric3x3<T> left, symmetric3x3<T> right) noexcept {
  using st = symmetric3x3<T>;
  return st(
      left(st::XX) - right(st::XX),
      left(st::YY) - right(st::YY),
      left(st::ZZ) - right(st::ZZ),
      left(st::XY) - right(st::XY),
      left(st::YZ) - right(st::YZ),
      left(st::XZ) - right(st::XZ));
}

template <class T>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE symmetric3x3<T>&
operator-=(symmetric3x3<T>& left, symmetric3x3<T> right) noexcept {
  left = left - right;
  return left;
}

template <class L, class R>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr symmetric3x3<L>
operator-(symmetric3x3<L> left, R right) noexcept {
  using st = symmetric3x3<L>;
  return st(
      left(st::XX) - right,
      left(st::YY) - right,
      left(st::ZZ) - right,
      left(st::XY),
      left(st::YZ),
      left(st::XZ));
}

template <class L, class R>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE symmetric3x3<L>&
operator-=(symmetric3x3<L>& left, R const right) noexcept {
  left = left - right;
  return left;
}

template <class L, class R>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr auto
operator*(symmetric3x3<L> const left, symmetric3x3<R> const right) noexcept {
  return left.full() * right.full();
}

template <class L, class R>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr auto
inner_product(symmetric3x3<L> left, symmetric3x3<R> right) noexcept {
  using st = symmetric3x3<L>;
  auto const diagonal_inner_product =
    (left(st::XX) * right(st::XX)) +
    (left(st::YY) * right(st::YY)) +
    (left(st::ZZ) * right(st::ZZ));
  auto const triangular_inner_product =
    (left(st::XY) * right(st::XY)) +
    (left(st::YZ) * right(st::YZ)) +
    (left(st::XZ) * right(st::XZ));
  return diagonal_inner_product + (2.0 * triangular_inner_product);
}

template <class L, class R>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr auto
operator*(matrix3x3<L> left, symmetric3x3<R> right) noexcept {
  return left * right.full();
}

template <class L, class R>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr auto
operator*(symmetric3x3<L> left, matrix3x3<R> right) noexcept {
  return left.full() * right;
}

template <class T>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr auto
self_times_transpose(matrix3x3<T> in) noexcept {
  return symmetric3x3<decltype(T() * T())>(in * transpose(in));
}

template <class T>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr auto
transpose_times_self(matrix3x3<T> in) noexcept {
  return symmetric3x3<decltype(T() * T())>(transpose(in) * in);
}

template <class L, class R>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr auto
operator*(symmetric3x3<L> left, vector3<R> right) noexcept {
  using out_t = vector3<decltype(L() * R())>;
  using st = decltype(left);
  return out_t(
      left(st::XX) * right(0) + left(st::XY) * right(1) + left(st::XZ) * right(2),
      left(st::XY) * right(0) + left(st::YY) * right(1) + left(st::YZ) * right(2),
      left(st::XZ) * right(0) + left(st::YZ) * right(1) + left(st::ZZ) * right(2));
}

template <class L, class R>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr auto
operator*(vector3<L> left, symmetric3x3<R> right) noexcept {
  return right * left;
}

template <class L, class R>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr auto
operator*(symmetric3x3<L> left, R right) noexcept {
  return symmetric3x3<decltype(L() * R())>(
      left(left.XX) * right,
      left(left.YY) * right,
      left(left.ZZ) * right,
      left(left.XY) * right,
      left(left.YZ) * right,
      left(left.XZ) * right);
}

template <class L, class R>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE symmetric3x3<L>&
operator*=(symmetric3x3<L>& left, R right) noexcept {
  left = left * right;
  return left;
}

template <class L, class R>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr auto
operator*(L left, symmetric3x3<R> right) noexcept {
  return right * left;
}

template <class L, class R>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr auto
operator/(symmetric3x3<L> left, R right) noexcept {
  using st = symmetric3x3<decltype(L() / R())>;
  auto const factor = 1.0 / right;
  return st(
      left(st::XX) * factor,
      left(st::YY) * factor,
      left(st::ZZ) * factor,
      left(st::XY) * factor,
      left(st::YZ) * factor,
      left(st::XZ) * factor);
}

template <class T>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr T
trace(symmetric3x3<T> x) noexcept {
  using st = symmetric3x3<T>;
  return x(st::XX) + x(st::YY) + x(st::ZZ);
}

template <class T>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr symmetric3x3<T>
deviator(symmetric3x3<T> x) noexcept {
  return x - ((1.0 / 3.0) * trace(x));
}

template <class T>
class array_traits<symmetric3x3<T>> {
  public:
  using value_type = T;
  HPC_HOST_DEVICE static constexpr std::ptrdiff_t size() noexcept { return 6; }
  template <class Iterator>
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE static symmetric3x3<T> load(Iterator it) noexcept {
    return symmetric3x3<T>(
        it[0],
        it[1],
        it[2],
        it[3],
        it[4],
        it[5]);
  }
  template <class Iterator>
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE static void store(Iterator it, symmetric3x3<T> const& value) noexcept {
    it[0] = value(symmetric3x3<T>::XX);
    it[1] = value(symmetric3x3<T>::YY);
    it[2] = value(symmetric3x3<T>::ZZ);
    it[3] = value(symmetric3x3<T>::XY);
    it[4] = value(symmetric3x3<T>::YZ);
    it[5] = value(symmetric3x3<T>::XZ);
  }
};

}

