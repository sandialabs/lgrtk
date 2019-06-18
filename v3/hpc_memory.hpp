#pragma once

#include <memory>
#include <type_traits>

#include <hpc_macros.hpp>
#include <hpc_execution.hpp>
#include <hpc_algorithm.hpp>

namespace hpc {

template <class T>
using host_allocator = std::allocator<T>;

#ifdef HPC_CUDA

template <class T>
class device_allocator {
 public:
  using value_type = T;
  using pointer = T*;
  using const_pointer = T const*;
  using reference = T&;
  using const_reference = T const&;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;
  template< class U > struct rebind { typedef ::hpc::device_allocator<U> other; };
  using is_always_equal = std::true_type;
  constexpr bool operator==(device_allocator const&) const noexcept { return true; }
  constexpr bool operator!=(device_allocator const&) const noexcept { return false; }
  T* allocate(std::size_t n) {
    auto err = cudaDeviceSynchronize();
    assert(err == cudaSuccess);
    void* ptr;
    err = cudaMalloc(&ptr, n * sizeof(T));
    if (err != cudaSuccess) {
      throw std::bad_alloc();
    }
    return static_cast<T*>(ptr);
  }
  void deallocate(T* p, std::size_t) {
    auto err = cudaDeviceSynchronize();
    (void)err;
    assert(err == cudaSuccess);
    err = cudaFree(p);
    (void)err;
    assert(cudaSuccess == err);
  }
};

template <class T>
class pinned_allocator {
 public:
  using value_type = T;
  using pointer = T*;
  using const_pointer = T const*;
  using reference = T&;
  using const_reference = T const&;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;
  template< class U > struct rebind { typedef ::hpc::pinned_allocator<U> other; };
  using is_always_equal = std::true_type;
  constexpr bool operator==(pinned_allocator const&) const noexcept { return true; }
  constexpr bool operator!=(pinned_allocator const&) const noexcept { return false; }
  T* allocate(std::size_t n) {
    auto err = cudaDeviceSynchronize();
    assert(err == cudaSuccess);
    void* ptr;
    err = cudaMallocHost(&ptr, n * sizeof(T));
    if (err != cudaSuccess) {
      throw std::bad_alloc();
    }
    return static_cast<T*>(ptr);
  }
  void deallocate(T* p, std::size_t) {
    auto err = cudaDeviceSynchronize();
    (void)err;
    assert(err == cudaSuccess);
    err = cudaFreeHost(p);
    (void)err;
    assert(cudaSuccess == err);
  }
};

#else

template <class T>
using pinned_allocator = std::allocator<T>;
template <class T>
using device_allocator = std::allocator<T>;

#endif

template <class Range>
HPC_NOINLINE void uninitialized_default_construct(serial_policy, Range&& range) {
  using range_type = std::decay_t<Range>;
  auto first = range.begin();
  auto const last = range.end();
  for (; first != last; ++first) {
    ::new (static_cast<void*>(std::addressof(*first))) typename range_type::value_type;
  }
}

#ifdef HPC_CUDA

template <class Range>
HPC_NOINLINE void uninitialized_default_construct(cuda_policy policy, Range&& range) {
  using range_type = std::decay_t<Range>;
  using reference_type = typename range_type::reference;
  auto functor = [=] HPC_DEVICE (reference_type ref) {
    ::new (static_cast<void*>(&ref)) typename range_type::value_type;
  };
  for_each(policy, range, functor);
}

#endif

template<class T>
HPC_ALWAYS_INLINE HPC_DEVICE void device_destroy_at(T* p)
{
  p->~T();
}

template<class T>
HPC_ALWAYS_INLINE void host_destroy_at(T* p)
{
  p->~T();
}

template <class Range>
HPC_NOINLINE void destroy(serial_policy, Range&& range) {
  auto first = range.begin();
  auto const last = range.end();
  for (; first != last; ++first) {
    ::hpc::host_destroy_at(std::addressof(*first));
  }
}

#ifdef HPC_CUDA

template <class Range>
HPC_NOINLINE void destroy(cuda_policy policy, Range&& range) {
  using range_type = std::decay_t<Range>;
  using reference_type = typename range_type::reference;
  auto functor = [=] HPC_DEVICE (reference_type ref) { 
    ::hpc::device_destroy_at(&ref);
  };
  ::hpc::for_each(policy, range, functor);
}

#endif

}
