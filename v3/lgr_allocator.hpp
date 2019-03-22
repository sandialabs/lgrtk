#pragma once

#include <type_traits>
#include <lgr_memory_pool.hpp>

namespace lgr {

template <class T>
struct concurrent_pooled_allocator {
  using value_type = T;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;
  template <class U> struct rebind { using other = concurrent_pooled_allocator<U>; };
  using is_always_equal = std::false_type;
  using propagate_on_container_swap = std::true_type;
  concurrent_memory_pool* m_pool;
  concurrent_pooled_allocator() = delete;
  explicit concurrent_pooled_allocator(concurrent_memory_pool& pool_in) noexcept;
  ~concurrent_pooled_allocator() noexcept = default;
  concurrent_pooled_allocator(concurrent_pooled_allocator const&) noexcept = default;
  concurrent_pooled_allocator& operator=(concurrent_pooled_allocator const&) noexcept = default;
  template <class U>
  concurrent_pooled_allocator(concurrent_pooled_allocator<U> const& other) noexcept
  :m_pool(other.m_pool)
  {}
  bool operator==(concurrent_pooled_allocator const&) noexcept;
  bool operator!=(concurrent_pooled_allocator const&) noexcept;
  T* allocate(std::size_t n);
  void deallocate(T* p, std::size_t n);
};

template <class T>
using host_allocator = std::allocator<T>;

template <class T>
struct device_allocator : public concurrent_pooled_allocator<T> {
  template <class U> struct rebind { using other = device_allocator<U>; };
  explicit device_allocator(device_memory_pool& pool_in) noexcept;
  template <class U>
  device_allocator(device_allocator<U> const& other) noexcept
  :concurrent_pooled_allocator<T>(other)
  {}
};

template <class T>
struct pinned_allocator : public concurrent_pooled_allocator<T> {
  template <class U> struct rebind { using other = pinned_allocator<U>; };
  explicit pinned_allocator(pinned_memory_pool& pool_in) noexcept;
  template <class U>
  pinned_allocator(pinned_allocator<U> const& other) noexcept
  :concurrent_pooled_allocator<T>(other)
  {}
};

}
