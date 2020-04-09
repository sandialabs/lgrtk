#pragma once

#include <hpc_macros.hpp>
#include <hpc_array_traits.hpp>
#include <hpc_index.hpp>
#include <hpc_math.hpp>
#include <hpc_matrix3x3.hpp>

namespace hpc {

using axis_index = hpc::index<axis_tag, int>;

template <typename Scalar>
class quaternion {
public:
  using scalar_type = Scalar;
private:
  scalar_type raw[4];
public:
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr quaternion(Scalar const w, Scalar const x, Scalar const y, Scalar const z) noexcept
    :raw{w, x, y, z}
  {
  }
  HPC_ALWAYS_INLINE quaternion() noexcept = default;
  HPC_ALWAYS_INLINE quaternion(quaternion<scalar_type> const&) noexcept = default;
  HPC_ALWAYS_INLINE quaternion& operator=(quaternion<scalar_type> const&) noexcept = default;
  template <class S2>
  HPC_ALWAYS_INLINE explicit quaternion(quaternion<S2> const& other) noexcept
    :quaternion(scalar_type(other(0)), scalar_type(other(1)), scalar_type(other(2)), scalar_type(other(3)))
  {}
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr scalar_type operator()(axis_index const i) const noexcept { return raw[weaken(i)]; }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE scalar_type& operator()(axis_index const i) noexcept { return raw[weaken(i)]; }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE static constexpr quaternion zero() noexcept { return quaternion(0.0, 0.0, 0.0, 0.0); }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE static constexpr quaternion w_axis() noexcept { return quaternion(1.0, 0.0, 0.0, 0.0); }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE static constexpr quaternion x_axis() noexcept { return quaternion(0.0, 1.0, 0.0, 0.0); }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE static constexpr quaternion y_axis() noexcept { return quaternion(0.0, 0.0, 1.0, 0.0); }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE static constexpr quaternion z_axis() noexcept { return quaternion(0.0, 0.0, 0.0, 1.0); }
};

template <class T>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr auto
operator+(quaternion<T> const left, quaternion<T> const right) noexcept {
  return quaternion<T>(
      left(0) + right(0),
      left(1) + right(1),
      left(2) + right(2),
      left(3) + right(3));
}

template <class T>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE quaternion<T>&
operator+=(quaternion<T>& left, quaternion<T> const right) noexcept {
  left = left + right;
  return left;
}

template <class T>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr auto
operator-(quaternion<T> const x) noexcept {
  return quaternion<T>(-x(0), -x(1), -x(2), -x(3));
}

template <class T>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr auto
operator-(quaternion<T> const left, quaternion<T> const right) noexcept {
  return quaternion<T>(
      left(0) - right(0),
      left(1) - right(1),
      left(2) - right(2),
      left(3) - right(3));
}

template <class T>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE quaternion<T>&
operator-=(quaternion<T>& left, quaternion<T> const right) noexcept {
  left = left - right;
  return left;
}

template <class L, class R>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr auto
inner_product(quaternion<L> const left, quaternion<R> const right) noexcept {
  return
      left(0) * right(0) +
      left(1) * right(1) +
      left(2) * right(2) +
      left(3) * right(3);
}

template <class L, class R>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr auto
operator*(quaternion<L> const left, quaternion<R> const right) noexcept {
  return inner_product(left, right);
}

template <class L, class R>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr auto
operator*(quaternion<L> const left, R const right) noexcept {
  return quaternion<decltype(L() * R())>(
      left(0) * right,
      left(1) * right,
      left(2) * right,
      left(3) * right);
}

template <class L, class R>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE quaternion<L>&
operator*=(quaternion<L>& left, R const right) noexcept {
  left = left * right;
  return left;
}

template <class L, class R>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr auto
operator*(L const left, quaternion<R> const right) noexcept {
  return right * left;
}

template <class L, class R>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr auto
operator/(quaternion<L> const left, R const right) noexcept {
  auto const factor = 1.0 / right;
  return quaternion<decltype(L() / R())>(
      left(0) * factor,
      left(1) * factor,
      left(2) * factor,
      left(3) * factor);
}

template <class L, class R>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE quaternion<L>&
operator/=(quaternion<L>& left, R const right) noexcept {
  left = left / right;
  return left;
}

template <class T>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr auto
norm_squared(quaternion<T> const v) noexcept {
  return (v * v);
}

template <class T>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE T norm(quaternion<T> const v) noexcept {
  using std::sqrt;
  return sqrt(norm_squared(v));
}

template <class T>
class array_traits<quaternion<T>> {
  public:
  using value_type = T;
  using size_type = axis_index;
  HPC_HOST_DEVICE static constexpr size_type size() noexcept { return 4; }
  template <class Iterator>
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE static quaternion<T> load(Iterator const it) noexcept {
    return quaternion<T>(
        it[0],
        it[1],
        it[2],
        it[3]);
  }
  template <class Iterator>
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE static void store(Iterator const it, quaternion<T> const& value) noexcept {
    it[0] = value(0);
    it[1] = value(1);
    it[2] = value(2);
    it[3] = value(3);
  }
};

template <typename T>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr auto
abs(quaternion<T> const v) noexcept {
  return quaternion<T>(std::abs(v(0)), std::abs(v(1)), std::abs(v(2)), std::abs(v(3)));
}

template <typename T>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr auto
normalize(quaternion<T> const v) noexcept {
  return v / norm(v);
}

//   Markley, F. Landis.
//   "Unit quaternion from rotation matrix."
//   Journal of guidance, control, and dynamics 31.2 (2008): 440-442.
//
//   Modified Shepperd's algorithm to handle input
//   tensors that may not be exactly orthogonal
template <typename T>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr auto
quaternion_from_rotation_tensor(matrix3x3<T> const R) noexcept {
  auto const trR = trace(R);
  auto maxm = trR;
  auto maxi = 3;
  auto q = quaternion<T>(0.0, 0.0, 0.0, 0.0);
  for (auto i = 0; i < 3; ++i) {
    if (R(i, i) > maxm) {
      maxm = R(i, i);
      maxi = i;
    }
  }
  if (maxi == 0) {
    q(1) = 1.0 + R(0, 0) - R(1, 1) - R(2, 2);
    q(2) = R(0, 1) + R(1, 0);
    q(3) = R(0, 2) + R(2, 0);
    q(0) = R(2, 1) - R(1, 2);
  } else if (maxi == 1) {
    q(1) = R(1, 0) + R(0, 1);
    q(2) = 1.0 + R(1, 1) - R(2, 2) - R(0, 0);
    q(3) = R(1, 2) + R(2, 1);
    q(0) = R(0, 2) - R(2, 0);
  } else if (maxi == 2) {
    q(1) = R(2, 0) + R(0, 2);
    q(2) = R(2, 1) + R(1, 2);
    q(3) = 1.0 + R(2, 2) - R(0, 0) - R(1, 1);
    q(0) = R(1, 0) - R(0, 1);
  } else if (maxi == 3) {
    q(1) = R(2, 1) - R(1, 2);
    q(2) = R(0, 2) - R(2, 0);
    q(3) = R(1, 0) - R(0, 1);
    q(0) = 1.0 + trR;
  }
  q = normalize(q);
  return q;
}

// This function maps a quaternion, q = (qs, qv), to its
// corresponding "principal" rotation pseudo-vector, a, where
// "principal" signifies that |a| <= π.  Both q and -q map into the
// same rotation matrix. It is convenient to require that qs >= 0, for
// reasons explained below.
//
//    |qv| = | sin(|a| / 2) |
//    qs  =   cos(|a| / 2)
//      <==>
//    |a| / 2 = k * π (+ or -) asin(|qv|)
//    |a| / 2 = 2 * l * π (+ or -) acos(qs)
//
// The smallest positive solution is:  |a| = 2 * acos(qs)
// which satisfies the inequality: 0 <= |a| <= π
// because of the assumption  qs >= 0.  Given |a|, a
// is obtained as:
//
//    a = (|a| / sin(acos(qs))) qv
//       = (|a| / sqrt(1 - qs^2)) qv
//
// The procedure described above is prone to numerical errors when qs
// is close to 1, i.e. when |a| is close to 0.  Since this is the most
// common case, special care must be taken.  It is observed that the
// cosine function is insensitive to perturbations of its argument in
// the neighborhood of points for which the sine function is conversely
// at its most sensitive.  Thus the numerical difficulties are avoided
// by computing |a| and a as:
//
//    |a| = 2 * asin(|qv|)
//    a  = (|a| / |qv|) qv
//
// whenever qs  is close to 1.

template <typename T>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr auto
rotation_vector_from_quaternion(quaternion<T> const q) noexcept {
  auto const qq = q(0) >= 0 ? q : -q;
  auto const qs = qq(0);
  auto const qv = vector3<T>(qq(1), qq(2), qq(3));
  auto const qvnorm = norm(qv);
  constexpr auto s = std::sqrt(0.5);
  constexpr auto e = std::sqrt(hpc::machine_epsilon<double>());
  auto const vnorm = 2.0 * (qvnorm < s ? std::asin(qvnorm) : std::acos(qs));
  auto const coef = qvnorm < e ? 2.0 : vnorm / qvnorm;
  auto const a = coef * qv;
  return a;
}

// In the algebra of rotations one often comes across functions that
// take undefined (0/0) values at some points.  Close to such points
// these functions must be evaluated using their asymptotic
// expansions; otherwise the computer may produce wildly erroneous
// results or a floating point exception.  To avoid unreadable code
// everywhere such functions are used, we introduce here functions to
// the same effect.
//
// NAME  FUNCTION FORM      X    ASSYMPTOTICS    FIRST RADIUS    SECOND RADIUS
// ----  -------------      -    ------------    ------------    -------------
// Ψ     sin(x)/x           0    1.0(-x^2/6)     (6*EPS)^.5      (120*EPS)^.25
template <typename T>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr auto
Psi(T const x) noexcept {
  auto const y = std::abs(x);
  constexpr auto e2 = std::sqrt(hpc::machine_epsilon<double>());
  constexpr auto e4 = std::sqrt(e2);
  auto const psi = y > e4 ? std::sin(y) / y : (y > e2 ? 1.0 - y * y / 6.0 : 1.0);
  return psi;
}

template <typename T>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr auto
quaternion_from_rotation_vector(vector3<T> const v) noexcept {
  auto const halfnorm = 0.5 * norm(v);
  auto const factor = 0.5 * Psi(halfnorm);
  auto const q = quaternion<T>(std::cos(halfnorm), factor * v(0), factor * v(1), factor * v(2));
  return q;
}

template <typename T>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr auto
rotation_tensor_from_quaternion(quaternion<T> const q) noexcept {
  auto const qs = q(0);
  auto const qv = vector3<T>(q(1), q(2), q(3));
  auto const I = matrix3x3<T>::identity();
  auto const R = 2.0 * outer_product(qv, qv) + 2.0 * qs * check(qv) + (2.0 * qs * qs - 1.0) * I;
  return R;
}

template <typename T>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr auto
rotation_vector_from_rotation_tensor(matrix3x3<T> const R) noexcept {
  auto const q = quaternion_from_rotation_tensor(R);
  auto const w = rotation_vector_from_quaternion(q);
  return w;
}

template <typename T>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr auto
rotation_tensor_from_rotation_vector(vector3<T> const w) noexcept {
  auto const q = quaternion_from_rotation_vector(w);
  auto const R = rotation_tensor_from_quaternion(q);
  return R;
}

} // namespace hpc
