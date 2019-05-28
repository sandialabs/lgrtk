#pragma once

#include <memory>
#include <type_traits>

#include <hpc_execution.hpp>

namespace hpc {

template <class T>
using host_allocator = std::allocator<T>;
template <class T>
using pinned_allocator = std::allocator<T>;
template <class T>
using device_allocator = std::allocator<T>;

template <class Range>
void uninitialized_default_construct(serial_policy, Range&& range) {
  using range_type = std::decay_t<Range>;
  auto first = range.begin();
  auto const last = range.end();
  for (; first != last; ++first) {
    ::new (static_cast<void*>(std::addressof(*first))) typename range_type::value_type;
  }
}

template<class T>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE void destroy_at(T* p)
{
  p->~T();
}

template <class Range>
void destroy(serial_policy, Range&& range) {
  auto first = range.begin();
  auto const last = range.end();
  for (; first != last; ++first) {
    ::hpc::destroy_at(std::addressof(*first));
  }
}

}
