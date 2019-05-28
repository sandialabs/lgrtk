#pragma once

#include <cmath>
#include <hpc_macros.hpp>
#include <hpc_array_traits.hpp>
#include <hpc_index.hpp>

namespace hpc {

struct axis_tag {};
using axis_index = hpc::index<axis_tag, int>;

template <typename Scalar>
class vector3 {
public:
  using scalar_type = Scalar;
private:
  scalar_type raw[3];
public:
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr inline vector3(Scalar const x, Scalar const y, Scalar const z) noexcept
    :raw{x, y, z}
  {
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE vector3() noexcept = default;
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr scalar_type operator()(axis_index i) const noexcept { return raw[i.get()]; }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE inline scalar_type& operator()(axis_index i) noexcept { return raw[i.get()]; }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE static constexpr vector3 zero() noexcept { return vector3(0.0, 0.0, 0.0); }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE static constexpr vector3 x_axis() noexcept { return vector3(1.0, 0.0, 0.0); }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE static constexpr vector3 y_axis() noexcept { return vector3(0.0, 1.0, 0.0); }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE static constexpr vector3 z_axis() noexcept { return vector3(0.0, 0.0, 1.0); }
};

template <class T>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr auto
operator+(vector3<T> left, vector3<T> right) noexcept {
  return vector3<T>(
      left(0) + right(0),
      left(1) + right(1),
      left(2) + right(2));
}

template <class T>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE vector3<T>&
operator+=(vector3<T>& left, vector3<T> right) noexcept {
  left = left + right;
  return left;
}

template <class T>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr auto
operator-(vector3<T> x) noexcept {
  return vector3<T>(-x(0), -x(1), -x(2));
}

template <class T>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr auto
operator-(vector3<T> left, vector3<T> right) noexcept {
  return vector3<T>(
      left(0) - right(0),
      left(1) - right(1),
      left(2) - right(2));
}

template <class T>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE vector3<T>&
operator-=(vector3<T>& left, vector3<T> right) noexcept {
  left = left - right;
  return left;
}

template <class L, class R>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr auto
inner_product(vector3<L> left, vector3<R> right) noexcept {
  return
      left(0) * right(0) +
      left(1) * right(1) +
      left(2) * right(2);
}

template <class L, class R>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr auto
operator*(vector3<L> left, vector3<R> right) noexcept {
  return inner_product(left, right);
}

template <class L, class R>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr auto
operator*(vector3<L> left, R right) noexcept {
  return vector3<decltype(L() * R())>(
      left(0) * right,
      left(1) * right,
      left(2) * right);
}

template <class L, class R>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE vector3<L>&
operator*=(vector3<L>& left, R right) noexcept {
  left = left * right;
  return left;
}

template <class L, class R>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr auto
operator*(L left, vector3<R> right) noexcept {
  return right * left;
}

template <class L, class R>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr auto
operator/(vector3<L> left, R right) noexcept {
  auto const factor = 1.0 / right;
  return vector3<decltype(L() / R())>(
      left(0) * factor,
      left(1) * factor,
      left(2) * factor);
}

template <class L, class R>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE vector3<L>&
operator/=(vector3<L>& left, R right) noexcept {
  left = left / right;
  return left;
}

template <class L, class R>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr auto
cross(vector3<L> left, vector3<R> right) noexcept {
  return vector3<decltype(L() * R())>(
      left(1) * right(2) - left(2) * right(1),
      left(2) * right(0) - left(0) * right(2),
      left(0) * right(1) - left(1) * right(0));
}

template <class T>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr auto
norm_squared(vector3<T> const v) noexcept {
  return (v * v);
}

template <class T>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE T norm(vector3<T> v) noexcept {
  using std::sqrt;
  return sqrt(norm_squared(v));
}

template <class T>
class array_traits<vector3<T>> {
  public:
  using value_type = T;
  using size_type = axis_index;
  HPC_HOST_DEVICE static constexpr size_type size() noexcept { return 3; }
  template <class Iterator>
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE static vector3<T> load(Iterator it) noexcept {
    return vector3<T>(
        it[0],
        it[1],
        it[2]);
  }
  template <class Iterator>
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE static void store(Iterator it, vector3<T> const& value) noexcept {
    it[0] = value(0);
    it[1] = value(1);
    it[2] = value(2);
  }
};

}
