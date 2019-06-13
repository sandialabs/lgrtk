#pragma once

#include <cmath>
#include <hpc_macros.hpp>
#include <hpc_array_traits.hpp>
#include <hpc_index.hpp>

namespace hpc {

struct axis_tag {};
using axis_index = hpc::index<axis_tag, int>;
static constexpr axis_index X{0};
static constexpr axis_index Y{1};
static constexpr axis_index Z{2};

template <typename Scalar>
class vector3 {
public:
  using scalar_type = Scalar;
private:
  scalar_type raw[3];
public:
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr vector3(Scalar const x, Scalar const y, Scalar const z) noexcept
    :raw{x, y, z}
  {
  }
  HPC_ALWAYS_INLINE vector3() noexcept = default;
  HPC_ALWAYS_INLINE vector3(vector3<scalar_type> const&) noexcept = default;
  HPC_ALWAYS_INLINE vector3& operator=(vector3<scalar_type> const&) noexcept = default;
  template <class S2>
  HPC_ALWAYS_INLINE explicit vector3(vector3<S2> const& other) noexcept
    :vector3(scalar_type(other(0)), scalar_type(other(1)), scalar_type(other(2)))
  {}
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr scalar_type operator()(axis_index const i) const noexcept { return raw[int(i)]; }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE scalar_type& operator()(axis_index const i) noexcept { return raw[int(i)]; }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE static constexpr vector3 zero() noexcept { return vector3(0.0, 0.0, 0.0); }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE static constexpr vector3 x_axis() noexcept { return vector3(1.0, 0.0, 0.0); }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE static constexpr vector3 y_axis() noexcept { return vector3(0.0, 1.0, 0.0); }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE static constexpr vector3 z_axis() noexcept { return vector3(0.0, 0.0, 1.0); }
};

template <class T>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr auto
operator+(vector3<T> const left, vector3<T> const right) noexcept {
  return vector3<T>(
      left(0) + right(0),
      left(1) + right(1),
      left(2) + right(2));
}

template <class T>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE vector3<T>&
operator+=(vector3<T>& left, vector3<T> const right) noexcept {
  left = left + right;
  return left;
}

template <class T>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr auto
operator-(vector3<T> const x) noexcept {
  return vector3<T>(-x(0), -x(1), -x(2));
}

template <class T>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr auto
operator-(vector3<T> const left, vector3<T> const right) noexcept {
  return vector3<T>(
      left(0) - right(0),
      left(1) - right(1),
      left(2) - right(2));
}

template <class T>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE vector3<T>&
operator-=(vector3<T>& left, vector3<T> const right) noexcept {
  left = left - right;
  return left;
}

template <class L, class R>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr auto
inner_product(vector3<L> const left, vector3<R> const right) noexcept {
  return
      left(0) * right(0) +
      left(1) * right(1) +
      left(2) * right(2);
}

template <class L, class R>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr auto
operator*(vector3<L> const left, vector3<R> const right) noexcept {
  return inner_product(left, right);
}

template <class L, class R>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr auto
operator*(vector3<L> const left, R const right) noexcept {
  return vector3<decltype(L() * R())>(
      left(0) * right,
      left(1) * right,
      left(2) * right);
}

template <class L, class R>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE vector3<L>&
operator*=(vector3<L>& left, R const right) noexcept {
  left = left * right;
  return left;
}

template <class L, class R>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr auto
operator*(L const left, vector3<R> const right) noexcept {
  return right * left;
}

template <class L, class R>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr auto
operator/(vector3<L> const left, R const right) noexcept {
  auto const factor = 1.0 / right;
  return vector3<decltype(L() / R())>(
      left(0) * factor,
      left(1) * factor,
      left(2) * factor);
}

template <class L, class R>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE vector3<L>&
operator/=(vector3<L>& left, R const right) noexcept {
  left = left / right;
  return left;
}

template <class L, class R>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr auto
cross(vector3<L> const left, vector3<R> const right) noexcept {
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
HPC_ALWAYS_INLINE HPC_HOST_DEVICE T norm(vector3<T> const v) noexcept {
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
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE static vector3<T> load(Iterator const it) noexcept {
    return vector3<T>(
        it[0],
        it[1],
        it[2]);
  }
  template <class Iterator>
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE static void store(Iterator const it, vector3<T> const& value) noexcept {
    it[0] = value(0);
    it[1] = value(1);
    it[2] = value(2);
  }
};

}
