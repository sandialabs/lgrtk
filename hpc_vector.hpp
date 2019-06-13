#pragma once

#include <hpc_memory.hpp>
#include <hpc_execution.hpp>
#include <hpc_iterator.hpp>
#include <hpc_algorithm.hpp>
#include <hpc_index.hpp>

namespace hpc {

template <
  class T,
  class Allocator = ::hpc::host_allocator<T>,
  class ExecutionPolicy = ::hpc::host_policy,
  class Index = std::ptrdiff_t>
class vector {
  using allocator_traits = std::allocator_traits<Allocator>;
  Allocator m_allocator;
  ExecutionPolicy m_execution_policy;
  T* m_data;
  Index m_size;
public:
  using value_type = T;
  using allocator_type = Allocator;
  using execution_policy = ExecutionPolicy;
  using size_type = Index;
  using difference_type = decltype(m_size - m_size);
  using reference = value_type&;
  using const_reference = value_type const&;
  using pointer = typename allocator_traits::pointer;
  using const_pointer = typename allocator_traits::const_pointer;
  using iterator = pointer_iterator<T, size_type>;
  using const_iterator = pointer_iterator<T const, size_type>;
  constexpr vector() noexcept
    :m_allocator()
    ,m_execution_policy()
    ,m_data(nullptr)
    ,m_size(0)
  {}
  vector(size_type count)
    :m_allocator()
    ,m_execution_policy()
    ,m_data(nullptr)
    ,m_size(0)
  {
    resize(count);
  }
  vector(size_type count, value_type const& value)
    :m_allocator()
    ,m_execution_policy()
    ,m_data(nullptr)
    ,m_size(0)
  {
    resize(count);
    ::hpc::fill(m_execution_policy, *this, value);
  }
  constexpr vector(allocator_type const& allocator_in, execution_policy const& exec_in) noexcept
    :m_allocator(allocator_in)
    ,m_execution_policy(exec_in)
    ,m_data(nullptr)
    ,m_size(0)
  {}
  vector(size_type count, allocator_type const& allocator_in, execution_policy const& exec_in)
    :m_allocator(allocator_in)
    ,m_execution_policy(exec_in)
    ,m_data(nullptr)
    ,m_size(0)
  {
    resize(count);
  }
  vector(vector&& other) noexcept
    :m_allocator(other.m_allocator)
    ,m_execution_policy(other.m_execution_policy)
    ,m_data(other.m_data)
    ,m_size(other.m_size)
  {
    other.m_data = nullptr;
    other.m_size = 0;
  }
  vector& operator=(vector&& other) {
    clear();
    m_allocator = other.m_allocator;
    m_execution_policy = other.m_execution_policy;
    m_data = other.m_data;
    m_size = other.m_size;
    other.m_data = nullptr;
    other.m_size = size_type(0);
    return *this;
  }
  vector(vector const&) = delete;
  vector& operator=(vector const&) = delete;
  ~vector() { clear(); }
  T* data() noexcept { return m_data; }
  T const* data() const noexcept { return m_data; }
  iterator begin() noexcept {
    return iterator(m_data, m_data, m_data + m_size);
  }
  const_iterator begin() const noexcept {
    return const_iterator(m_data, m_data, m_data + m_size);
  }
  const_iterator cbegin() const noexcept {
    return const_iterator(m_data, m_data, m_data + m_size);
  }
  iterator end() noexcept {
    return iterator(m_data + m_size, m_data, m_data + m_size);
  }
  const_iterator end() const noexcept {
    return const_iterator(m_data + m_size, m_data, m_data + m_size);
  }
  const_iterator cend() const noexcept {
    return const_iterator(m_data + m_size, m_data, m_data + m_size);
  }
  bool empty() const noexcept {
    return m_size == 0;
  }
  size_type size() const noexcept { return m_size; }
  void clear() {
    if (m_data) {
      if (!std::is_trivially_destructible<value_type>::value) {
        ::hpc::destroy(m_execution_policy, *this);
      }
      allocator_traits::deallocate(m_allocator, m_data, std::size_t(hpc::weaken(m_size)));
    }
    m_data = nullptr;
    m_size = size_type(0);
  }
  void resize(size_type count) {
    if (m_size == count) return;
    if (count == size_type(0)) {
      clear();
      return;
    }
    if (m_size == size_type(0)) {
      clear();
      m_data = allocator_traits::allocate(m_allocator, std::size_t(weaken(count)));
      m_size = count;
      if (!std::is_trivially_constructible<T>::value) {
        ::hpc::uninitialized_default_construct(m_execution_policy, *this);
      }
      return;
    }
    auto const mid = ::hpc::min(m_size, count);
    auto const move_from_range = ::hpc::make_iterator_range(begin(), begin() + mid);
    auto const new_data = allocator_traits::allocate(m_allocator, std::size_t(weaken(count)));
    iterator const new_begin(new_data, new_data, new_data + count);
    auto const move_into_range = ::hpc::make_iterator_range(new_begin, new_begin + mid);
    ::hpc::move(m_execution_policy, move_from_range, move_into_range);
    clear();
    if (!std::is_trivially_constructible<T>::value) {
      auto const construct_range = ::hpc::make_iterator_range(new_begin + mid, new_begin + count);
      ::hpc::uninitialized_default_construct(m_execution_policy, construct_range);
    }
    m_data = new_data;
    m_size = count;
  }
  constexpr allocator_type get_allocator() const noexcept { return m_allocator; }
  constexpr execution_policy get_execution_policy() const noexcept { return m_execution_policy; }
  constexpr reference operator[](size_type i) noexcept { return begin()[i]; }
  constexpr const_reference operator[](size_type i) const noexcept { return begin()[i]; }
};

template <class T, class Index = std::ptrdiff_t>
using host_vector = vector<T, ::hpc::host_allocator<T>, ::hpc::host_policy, Index>;
template <class T, class Index = std::ptrdiff_t>
using device_vector = vector<T, ::hpc::device_allocator<T>, ::hpc::device_policy, Index>;
template <class T, class Index = std::ptrdiff_t>
using pinned_vector = vector<T, ::hpc::pinned_allocator<T>, ::hpc::host_policy, Index>;

template <class T, class A, class P, class I>
void copy(vector<T, A, P, I> const& from, vector<T, A, P, I>& to) {
  hpc::copy(from.get_execution_policy(), from, to);
}

#ifdef HPC_CUDA

template <class T, class Index>
void copy(pinned_vector<T, Index> const& from, device_vector<T, Index>& to) {
  assert(from.size() == to.size());
  std::size_t const size = std::size_t(from.size());
  auto const from_ptr = from.data();
  auto const to_ptr = to.data();
  auto err = cudaMemcpy(to_ptr, from_ptr, size * sizeof(T), cudaMemcpyHostToDevice);
  assert(cudaSuccess == err);
}

template <class T, class Index>
void copy(device_vector<T, Index> const& from, pinned_vector<T, Index>& to) {
  assert(from.size() == to.size());
  std::size_t const size = std::size_t(to.size());
  auto const from_ptr = from.data();
  auto const to_ptr = to.data();
  auto err = cudaMemcpy(to_ptr, from_ptr, size * sizeof(T), cudaMemcpyDeviceToHost);
  assert(cudaSuccess == err);
}

#endif

}
