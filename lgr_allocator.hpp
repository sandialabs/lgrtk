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
  memory_pool_base* m_pool;
  concurrent_pooled_allocator() = delete;
  explicit concurrent_pooled_allocator(memory_pool_base& pool_in) noexcept
    :m_pool(&pool_in)
  {
  }
  ~concurrent_pooled_allocator() noexcept = default;
  concurrent_pooled_allocator(concurrent_pooled_allocator const&) noexcept = default;
  concurrent_pooled_allocator& operator=(concurrent_pooled_allocator const&) noexcept = default;
  template <class U>
  concurrent_pooled_allocator(concurrent_pooled_allocator<U> const& other) noexcept
  :m_pool(other.m_pool)
  {}
  bool operator==(concurrent_pooled_allocator const& other) noexcept
  {
    return m_pool == other.m_pool;
  }
  bool operator!=(concurrent_pooled_allocator const& other) noexcept
  {
    return m_pool != other.m_pool;
  }
  T* allocate(std::size_t n) {
    return static_cast<T*>(m_pool->allocate(n * sizeof(T)));
  }
  void deallocate(T* p, std::size_t n) {
    m_pool->deallocate(p, n * sizeof(T));
  }
  memory_pool_base& get_pool() { return *m_pool; }
};

template <class T>
using host_allocator = std::allocator<T>;

template <class T>
struct device_allocator : public concurrent_pooled_allocator<T> {
  template <class U> struct rebind { using other = device_allocator<U>; };
  explicit device_allocator(device_memory_pool& pool_in) noexcept
    :concurrent_pooled_allocator<T>(pool_in)
  {
  }
  template <class U>
  device_allocator(device_allocator<U> const& other) noexcept
  :concurrent_pooled_allocator<T>(other)
  {}
  device_memory_pool& get_pool() { return *static_cast<device_memory_pool*>(this->m_pool); }
};

template <class T>
struct pinned_allocator : public concurrent_pooled_allocator<T> {
  template <class U> struct rebind { using other = pinned_allocator<U>; };
  explicit pinned_allocator(pinned_memory_pool& pool_in) noexcept
    :concurrent_pooled_allocator<T>(pool_in)
  {
  }
  template <class U>
  pinned_allocator(pinned_allocator<U> const& other) noexcept
  :concurrent_pooled_allocator<T>(other)
  {}
};

}
