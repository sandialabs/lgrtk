#pragma once

#include <lgr_vector.hpp>

namespace lgr {

template <class T, class Index = int>
class host_vector {
  using impl_type = vector<T, host_allocator<T>, Index>;
  impl_type m_impl;
public:
  using typename impl_type::value_type;
  using typename impl_type::allocator_type;
  using typename impl_type::size_type;
  using typename impl_type::difference_type;
  using typename impl_type::reference;
  using typename impl_type::const_reference;
  using typename impl_type::pointer;
  using typename impl_type::const_pointer;
  using typename impl_type::iterator;
  using typename impl_type::const_iterator;
  explicit host_vector() noexcept
    :m_impl(host_allocator<T>())
  {}
  template <class ... Args>
  explicit host_vector(size_type count, Args ... args)
    :m_impl(host_allocator<T>())
  {
    resize(count, args...);
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
  void resize(size_type count, Args ... args) {
    clear();
    m_impl.resize(count);
    T* const ptr = data();
    for (size_type i(0); i < count; ++i) {
      new (ptr + int(i)) T(args...);
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

