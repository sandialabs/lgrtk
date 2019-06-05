#pragma once

#include <hpc_macros.hpp>

namespace hpc {

template <class T>
class atomic_ref;

template <>
class atomic_ref<int> {
  public:
  using value_type = int;
  private:
  value_type& m_ref;
  public:
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE explicit atomic_ref(value_type& ref_in) noexcept : m_ref(ref_in) {}
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE value_type operator++(int) const noexcept {
#ifdef __CUDA_ARCH__
    return atomicAdd(&m_ref, 1);
#else
    return m_ref++;
#endif
  }
};

}
