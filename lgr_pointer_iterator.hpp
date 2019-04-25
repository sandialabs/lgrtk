#pragma once

#ifndef NDEBUG
#include <cassert>
#endif

namespace lgr {

template <class T, class Index = int>
class pointer_iterator {
  T* m_ptr;
#ifndef NDEBUG
  T* m_begin;
  T* m_end;
#endif
 public:
  using value_type = std::remove_const_t<T>;
  using difference_type = decltype(Index(0) - Index(0));
  using reference = T&;
  using pointer = T*;
  using iterator_category = std::random_access_iterator_tag;
#ifdef NDEBUG
  explicit inline pointer_iterator(T* const ptr_in) noexcept : m_ptr(ptr_in) {}
#else
  explicit inline pointer_iterator(T* const ptr_in, T* const begin_in, T* const end_in) noexcept
    : m_ptr(ptr_in), m_begin(begin_in), m_end(end_in) {}
#endif
  inline bool operator==(pointer_iterator const& other) const noexcept {
    return m_ptr == other.m_ptr;
  }
  inline bool operator!=(pointer_iterator const& other) const noexcept {
    return m_ptr != other.m_ptr;
  }
  inline reference operator*() const noexcept {
#ifndef NDEBUG
    assert(m_begin <= m_ptr);
    assert(m_ptr < m_end);
#endif
    return *m_ptr;
  }
  inline pointer_iterator& operator++() noexcept {
    ++m_ptr;
    return *this;
  }
  inline pointer_iterator operator++(int) noexcept {
    auto ret = *this;
    ++m_ptr;
    return ret;
  }
  inline pointer_iterator& operator--() noexcept {
    --m_ptr;
    return *this;
  }
  inline pointer_iterator operator--(int) noexcept {
    auto ret = *this;
    --m_ptr;
    return ret;
  }
  inline pointer_iterator& operator+=(difference_type const n) noexcept {
    m_ptr += int(n);
    return *this;
  }
  inline pointer_iterator& operator-=(difference_type const n) noexcept {
    m_ptr -= int(n);
    return *this;
  }
  inline pointer_iterator operator+(difference_type const n) const noexcept {
#ifdef NDEBUG
    return pointer_iterator(m_ptr + int(n));
#else
    return pointer_iterator(m_ptr + int(n), m_begin, m_end);
#endif
  }
  inline pointer_iterator operator-(difference_type const n) const noexcept {
#ifdef NDEBUG
    return pointer_iterator(m_ptr - int(n));
#else
    return pointer_iterator(m_ptr - int(n), m_begin, m_end);
#endif
  }
  inline difference_type operator-(pointer_iterator const& other) const noexcept {
    return difference_type(m_ptr - other.m_ptr);
  }
  inline reference operator[](Index const i) const noexcept {
    return *((*this) + Index(i));
  }
  inline bool operator<(pointer_iterator const& other) const noexcept {
    return m_ptr < other.m_ptr;
  }
  inline bool operator>(pointer_iterator const& other) const noexcept {
    return m_ptr > other.m_ptr;
  }
  inline bool operator<=(pointer_iterator const& other) const noexcept {
    return m_ptr <= other.m_ptr;
  }
  inline bool operator>=(pointer_iterator const& other) const noexcept {
    return m_ptr >= other.m_ptr;
  }
};

}
