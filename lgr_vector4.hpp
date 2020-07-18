#pragma once

#include <hpc_macros.hpp>

namespace lgr {

template <typename Scalar>
class vector4
{
 public:
  using scalar_type = Scalar;

 private:
  scalar_type raw[4];

 public:
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE explicit constexpr vector4(
      Scalar const x,
      Scalar const y,
      Scalar const z,
      Scalar const w) noexcept
      : raw{x, y, z, w}
  {
  }
  HPC_ALWAYS_INLINE
  vector4() noexcept = default;
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr scalar_type
  operator()(int const i) const noexcept
  {
    return raw[i];
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE scalar_type&
                                    operator()(int const i) noexcept
  {
    return raw[i];
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE static constexpr vector4
  zero() noexcept
  {
    return vector4(0.0, 0.0, 0.0, 0.0);
  }
};

template <typename Scalar>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr Scalar
inner_product(vector4<Scalar> const left, vector4<Scalar> const right) noexcept
{
  return left(0) * right(0) + left(1) * right(1) + left(2) * right(2) +
         left(3) * right(3);
}

template <typename Scalar>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr Scalar
operator*(vector4<Scalar> const left, vector4<Scalar> const right) noexcept
{
  return inner_product(left, right);
}

}  // namespace lgr
