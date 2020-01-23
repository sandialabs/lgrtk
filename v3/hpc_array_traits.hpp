#pragma once

#include <cfloat>
#include <cstddef>
#include <hpc_macros.hpp>
#include <limits>

namespace hpc {

// specialize this to describe how T is convertible to an array
template <class T>
class array_traits {
  public:
  using value_type = T;
  using size_type = std::ptrdiff_t;
  HPC_HOST_DEVICE static constexpr size_type size() noexcept { return 1; }
  template <class Iterator>
  HPC_HOST_DEVICE static T load(Iterator it) noexcept { return *it; }
  template <class Iterator>
  HPC_HOST_DEVICE static void store(Iterator it, T const& value) noexcept { *it = value; }
};

// machine epsilon
template <typename T>
HPC_HOST_DEVICE constexpr auto
machine_epsilon()
{
  return std::numeric_limits<T>::epsilon();
}

template <>
HPC_HOST_DEVICE  constexpr auto
machine_epsilon<double>()
{
  return DBL_EPSILON;
}

}
