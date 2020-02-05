#pragma once

#include <hpc_macros.hpp>

namespace hpc {

template <class T>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr T max(T const& a, T const& b) noexcept {
  return (a < b) ? b : a;
}

template<class T>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr T min(T const& a, T const& b) noexcept {
  return (b < a) ? b : a;
}

template <class T>
struct minimum {
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr T operator()(T const& a, T const& b) noexcept {
    return ::hpc::min(a, b);
  }
};

template <class T>
struct maximum {
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr T operator()(T const& a, T const& b) noexcept {
    return ::hpc::max(a, b);
  }
};

template <class T>
struct plus {
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr T operator()(T const& a, T const& b) noexcept {
    return a + b;
  }
};

template <class T>
struct identity {
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr T operator()(T const& a) noexcept {
    return a;
  }
};

struct logical_or
{
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr bool operator()(bool const a, bool const b) const noexcept { return a || b; }
};

struct logical_and
{
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr bool operator()(bool const a, bool const b) const noexcept { return a && b; }
};

template <class To, class From>
struct cast {
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr To operator()(From const& a) noexcept {
    return a;
  }
};

}
