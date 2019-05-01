#pragma once

#include <lgr_vector.hpp>
#include <lgr_allocator.hpp>

namespace lgr {

template <class T, class Index = int>
class host_vector {
  using impl_type = vector<T, host_allocator<T>, Index>;
  impl_type m_impl;
public:
  using value_type = typename impl_type::value_type;
  using allocator_type = typename impl_type::allocator_type;
  using size_type = typename impl_type::size_type;
  using difference_type = typename impl_type::difference_type;
  using reference = typename impl_type::reference;
  using const_reference = typename impl_type::const_reference;
  using pointer = typename impl_type::pointer;
  using const_pointer = typename impl_type::const_pointer;
  using iterator = typename impl_type::iterator;
  using const_iterator = typename impl_type::const_iterator;
  explicit host_vector() noexcept
    :m_impl(host_allocator<T>())
  {}
  template <class ... Args>
  explicit host_vector(size_type count, Args&& ... args)
    :m_impl(host_allocator<T>())
  {
    resize(count, args...);
  }
  explicit host_vector(size_type count)
    :m_impl(host_allocator<T>())
  {
    resize(count);
  }
  host_vector(host_vector&&) = delete;
  host_vector(host_vector const&) = delete;
  host_vector& operator=(host_vector const&) = delete;
  host_vector& operator=(host_vector&&) = delete;
  ~host_vector() {
    clear();
  }
  T* data() noexcept { return m_impl.data(); }
  T const* data() const noexcept { return m_impl.data(); }
  iterator begin() noexcept { return m_impl.begin(); }
  const_iterator begin() const noexcept { return m_impl.begin(); }
  const_iterator cbegin() const noexcept { return m_impl.cbegin(); }
  iterator end() noexcept { return m_impl.end(); }
  const_iterator end() const noexcept { return m_impl.end(); }
  const_iterator cend() const noexcept { return m_impl.cend(); }
  bool empty() const noexcept { return m_impl.empty(); }
  size_type size() const noexcept { return m_impl.size(); }
  reference operator[](Index const i) { return m_impl.begin()[i]; }
  const_reference operator[](Index const i) const { return m_impl.cbegin()[i]; }
  template <class ... Args>
  void resize(size_type count, Args&& ... args) {
    clear();
    m_impl.resize(count);
    T* const ptr = data();
    for (size_type i(0); i < count; ++i) {
      new (ptr + int(i)) T(args...);
    }
  }
  void resize(size_type count) {
    clear();
    m_impl.resize(count);
    T* const ptr = data();
    for (size_type i(0); i < count; ++i) {
      new (ptr + int(i)) T();
    }
  }
  void clear() {
    size_type const count = size();
    T* const ptr = data();
    for (size_type i(0); i < count; ++i) {
      ptr[int(i)].~T();
    }
    m_impl.clear();
  }
  allocator_type get_allocator() const { return m_impl.get_allocator(); }
};

}
