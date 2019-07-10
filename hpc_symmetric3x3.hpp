#pragma once

#include <hpc_matrix3x3.hpp>

namespace hpc {

class symmetric_tag {};
using symmetric_index = index<symmetric_tag, int>;
static constexpr symmetric_index S_XX{0};
static constexpr symmetric_index S_YY{1};
static constexpr symmetric_index S_ZZ{2};
static constexpr symmetric_index S_XY{3};
static constexpr symmetric_index S_YZ{4};
static constexpr symmetric_index S_XZ{5};

template <typename Scalar>
struct symmetric3x3 {
public:
  using scalar_type = Scalar;
private:
  scalar_type raw[6];
public:
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr symmetric3x3(
      Scalar const a, Scalar const b, Scalar const c,
      Scalar const d, Scalar const e, Scalar const f,
      Scalar const g, Scalar const h, Scalar const i) noexcept
  {
    operator()(S_XX) = a;
    operator()(S_YY) = e;
    operator()(S_ZZ) = i;
    operator()(S_XY) = (b + d) * 0.5;
    operator()(S_YZ) = (f + h) * 0.5;
    operator()(S_XZ) = (c + g) * 0.5;
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr symmetric3x3(
      Scalar const xx, Scalar const yy, Scalar const zz,
      Scalar const xy, Scalar const yz, Scalar const xz) noexcept
  {
    operator()(S_XX) = xx;
    operator()(S_YY) = yy;
    operator()(S_ZZ) = zz;
    operator()(S_XY) = xy;
    operator()(S_YZ) = yz;
    operator()(S_XZ) = xz;
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE explicit constexpr symmetric3x3(matrix3x3<Scalar> const t):
    symmetric3x3(
        t(0, 0), t(0, 1), t(0, 2),
        t(1, 0), t(1, 1), t(1, 2),
        t(2, 0), t(2, 1), t(2, 2))
  {
  }
  HPC_ALWAYS_INLINE symmetric3x3() = default;
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE static constexpr symmetric3x3 identity() noexcept {
    return symmetric3x3(1.0, 1.0, 1.0, 0.0, 0.0, 0.0);
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr matrix3x3<Scalar> full() const noexcept {
    return matrix3x3<Scalar>(
        operator()(S_XX), operator()(S_XY), operator()(S_XZ),
        operator()(S_XY), operator()(S_YY), operator()(S_YZ),
        operator()(S_XZ), operator()(S_YZ), operator()(S_ZZ));
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr Scalar& operator()(symmetric_index const slot) noexcept {
    return raw[weaken(slot)];
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr Scalar operator()(symmetric_index const slot) const noexcept {
    return raw[weaken(slot)];
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE static constexpr symmetric3x3<Scalar> zero() noexcept {
    return symmetric3x3<Scalar>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE Scalar& operator()(axis_index const i, axis_index const j) noexcept {
    if (i == 0 && j == 0) return operator()(S_XX);
    if (i == 0 && j == 1) return operator()(S_XY);
    if (i == 0 && j == 2) return operator()(S_XZ);
    if (i == 1 && j == 0) return operator()(S_XY);
    if (i == 1 && j == 1) return operator()(S_YY);
    if (i == 1 && j == 2) return operator()(S_YZ);
    if (i == 2 && j == 0) return operator()(S_XZ);
    if (i == 2 && j == 1) return operator()(S_YZ);
    return operator()(S_ZZ);
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE Scalar operator()(axis_index const i, axis_index const j) const noexcept {
    if (i == 0 && j == 0) return operator()(S_XX);
    if (i == 0 && j == 1) return operator()(S_XY);
    if (i == 0 && j == 2) return operator()(S_XZ);
    if (i == 1 && j == 0) return operator()(S_XY);
    if (i == 1 && j == 1) return operator()(S_YY);
    if (i == 1 && j == 2) return operator()(S_YZ);
    if (i == 2 && j == 0) return operator()(S_XZ);
    if (i == 2 && j == 1) return operator()(S_YZ);
    return operator()(S_ZZ);
  }
};

template <class T>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr symmetric3x3<T>
operator+(symmetric3x3<T> const left, symmetric3x3<T> const right) noexcept {
  return symmetric3x3<T>(
      left(0) + right(0),
      left(1) + right(1),
      left(2) + right(2),
      left(3) + right(3),
      left(4) + right(4),
      left(5) + right(5));
}

template <class T>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE symmetric3x3<T>&
operator+=(symmetric3x3<T>& left, symmetric3x3<T> const right) noexcept {
  left = left + right;
  return left;
}

template <class L, class R>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr symmetric3x3<L>
operator+(symmetric3x3<L> const left, R const right) noexcept {
  return symmetric3x3<L>(
      left(0) + right,
      left(1) + right,
      left(2) + right,
      left(3),
      left(4),
      left(5));
}

template <class L, class R>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE symmetric3x3<L>&
operator+=(symmetric3x3<L>& left, R const right) noexcept {
  left = left + right;
  return left;
}

template <class L, class R>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr auto
operator+(L const left, symmetric3x3<R> const right) noexcept {
  return right + left;
}

template <class T>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr symmetric3x3<T>
operator-(symmetric3x3<T> const left, symmetric3x3<T> const right) noexcept {
  using st = symmetric3x3<T>;
  return st(
      left(0) - right(0),
      left(1) - right(1),
      left(2) - right(2),
      left(3) - right(3),
      left(4) - right(4),
      left(5) - right(5));
}

template <class T>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE symmetric3x3<T>&
operator-=(symmetric3x3<T>& left, symmetric3x3<T> const right) noexcept {
  left = left - right;
  return left;
}

template <class L, class R>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr symmetric3x3<L>
operator-(symmetric3x3<L> const left, R const right) noexcept {
  using st = symmetric3x3<L>;
  return st(
      left(0) - right,
      left(1) - right,
      left(2) - right,
      left(3),
      left(4),
      left(5));
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
inner_product(symmetric3x3<L> const left, symmetric3x3<R> const right) noexcept {
  auto const diagonal_inner_product =
    (left(S_XX) * right(S_XX)) +
    (left(S_YY) * right(S_YY)) +
    (left(S_ZZ) * right(S_ZZ));
  auto const triangular_inner_product =
    (left(S_XY) * right(S_XY)) +
    (left(S_YZ) * right(S_YZ)) +
    (left(S_XZ) * right(S_XZ));
  return diagonal_inner_product + (2.0 * triangular_inner_product);
}

template <class L, class R>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr auto
operator*(matrix3x3<L> const left, symmetric3x3<R> const right) noexcept {
  return left * right.full();
}

template <class L, class R>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr auto
operator*(symmetric3x3<L> const left, matrix3x3<R> const right) noexcept {
  return left.full() * right;
}

template <class T>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr auto
self_times_transpose(matrix3x3<T> const in) noexcept {
  return symmetric3x3<decltype(T() * T())>(in * transpose(in));
}

template <class T>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr auto
transpose_times_self(matrix3x3<T> const in) noexcept {
  return symmetric3x3<decltype(T() * T())>(transpose(in) * in);
}

template <class L, class R>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr auto
operator*(symmetric3x3<L> const left, vector3<R> const right) noexcept {
  return vector3<decltype(L() * R())>(
      left(S_XX) * right(X) + left(S_XY) * right(Y) + left(S_XZ) * right(Z),
      left(S_XY) * right(X) + left(S_YY) * right(Y) + left(S_YZ) * right(Z),
      left(S_XZ) * right(X) + left(S_YZ) * right(Y) + left(S_ZZ) * right(Z));
}

template <class L, class R>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr auto
operator*(vector3<L> const left, symmetric3x3<R> const right) noexcept {
  return right * left;
}

template <class L, class R>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr auto
operator*(symmetric3x3<L> const left, R const right) noexcept {
  return symmetric3x3<decltype(L() * R())>(
      left(0) * right,
      left(1) * right,
      left(2) * right,
      left(3) * right,
      left(4) * right,
      left(5) * right);
}

template <class L, class R>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE symmetric3x3<L>&
operator*=(symmetric3x3<L>& left, R const right) noexcept {
  left = left * right;
  return left;
}

template <class L, class R>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr auto
operator*(L const left, symmetric3x3<R> const right) noexcept {
  return right * left;
}

template <class L, class R>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr auto
operator/(symmetric3x3<L> const left, R const right) noexcept {
  using st = symmetric3x3<decltype(L() / R())>;
  auto const factor = 1.0 / right;
  return st(
      left(0) * factor,
      left(1) * factor,
      left(2) * factor,
      left(3) * factor,
      left(4) * factor,
      left(5) * factor);
}

template <class T>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr T
trace(symmetric3x3<T> const x) noexcept {
  return x(S_XX) + x(S_YY) + x(S_ZZ);
}

template <class T>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr symmetric3x3<T>
symmetric_part(matrix3x3<T> const x) noexcept {
  // The constructor that takes a full matrix implicity takes its symmetric part
  return symmetric3x3<T>(x);
}

template <class T>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr symmetric3x3<T>
isotropic_part(symmetric3x3<T> const x) noexcept {
  return ((1.0 / 3.0) * trace(x)) * symmetric3x3<T>::identity();
}

template <class T>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr symmetric3x3<T>
deviator(symmetric3x3<T> const x) noexcept {
  auto x_dev = symmetric3x3<T>(x);
  auto const a = (1.0 / 3.0) * trace(x);
  x_dev(S_XX) -= a;
  x_dev(S_YY) -= a;
  x_dev(S_YY) -= a;
  return x_dev;
}

template <class T>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr T
norm(symmetric3x3<T> const x) noexcept {
  T norm_sq = x(S_XX) * x(S_XX) + x(S_YY) * x(S_YY) + x(S_ZZ) * x(S_ZZ)
            + 2.0 * (x(S_XY) * x(S_XY) + x(S_YZ) * x(S_YZ) + x(S_XZ) * x(S_XZ));
  return std::sqrt(norm_sq);
}

template <typename T>
HPC_HOST_DEVICE constexpr auto
determinant(symmetric3x3<T> const x) noexcept {
  T const xx = x(S_XX); T const xy = x(S_XY); T const xz = x(S_XZ);
                        T const yy = x(S_YY); T const yz = x(S_YZ);
                                              T const zz = x(S_ZZ);
  return (xx * yy * zz) - (xx * yz * yz) - (xy * xy * zz) +
          2.0 * (xy * xz * yz) - (xz * xz * yy);
}

template <class T>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr symmetric3x3<T>
inverse(symmetric3x3<T> const x) noexcept {
  T const xx = x(S_XX); T const xy = x(S_XY); T const xz = x(S_XZ);
                        T const yy = x(S_YY); T const yz = x(S_YZ);
                                              T const zz = x(S_ZZ);
  auto const A =  (yy * zz - yz * yz);
  auto const B =  (xx * zz - xz * xz);
  auto const C =  (xx * yy - xy * xy);
  auto const D = -(xy * zz - xz * yz);
  auto const E = -(xx * yz - xz * xy);
  auto const F =  (xy * yz - xz * yy);
  using top_t = symmetric3x3<std::remove_const_t<decltype(A)>>;
  auto const top = top_t(A, B, C, D, E, F);
  return top / determinant(x);
}

template <class T>
class array_traits<symmetric3x3<T>> {
  public:
  using value_type = T;
  using size_type = symmetric_index;
  HPC_HOST_DEVICE static constexpr size_type size() noexcept { return 6; }
  template <class Iterator>
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE static symmetric3x3<T> load(Iterator const it) noexcept {
    return symmetric3x3<T>(
        it[0],
        it[1],
        it[2],
        it[3],
        it[4],
        it[5]);
  }
  template <class Iterator>
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE static void store(Iterator const it, symmetric3x3<T> const& value) noexcept {
    it[0] = value(0);
    it[1] = value(1);
    it[2] = value(2);
    it[3] = value(3);
    it[4] = value(4);
    it[5] = value(5);
  }
};

}

