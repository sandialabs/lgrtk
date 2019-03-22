#pragma once

#include <lgr_int_range.hpp>
#include <lgr_vector.hpp>

namespace lgr {

class int_range_sum_iterator {
  int const* m_ptr;

 public:
  using value_type = int_range;
  using difference_type = int;
  using reference = value_type;
  using pointer = value_type const*;
  using iterator_category = std::random_access_iterator_tag;
  explicit inline int_range_sum_iterator(int const* ptr_in) noexcept : m_ptr(ptr_in) {}
  inline bool operator==(int_range_sum_iterator const& other) const noexcept {
    return m_ptr == other.m_ptr;
  }
  inline bool operator!=(int_range_sum_iterator const& other) const noexcept {
    return m_ptr != other.m_ptr;
  }
  inline reference operator*() const noexcept { return int_range(m_ptr[0], m_ptr[1]); }
  inline int_range_sum_iterator& operator++() noexcept {
    ++m_ptr;
    return *this;
  }
  inline int_range_sum_iterator operator++(int) noexcept {
    auto ret = *this;
    ++m_ptr;
    return ret;
  }
  inline int_range_sum_iterator& operator--() noexcept {
    --m_ptr;
    return *this;
  }
  inline int_range_sum_iterator operator--(int) noexcept {
    auto ret = *this;
    --m_ptr;
    return ret;
  }
  inline int_range_sum_iterator& operator+=(difference_type const n) noexcept {
    m_ptr += n;
    return *this;
  }
  inline int_range_sum_iterator& operator-=(difference_type const n) noexcept {
    m_ptr -= n;
    return *this;
  }
  inline int_range_sum_iterator operator+(difference_type const n) const noexcept {
    return int_range_sum_iterator(m_ptr + n);
  }
  inline int_range_sum_iterator operator-(difference_type const n) const noexcept {
    return int_range_sum_iterator(m_ptr - n);
  }
  inline difference_type operator-(int_range_sum_iterator const& other) const noexcept {
    return difference_type(m_ptr - other.m_ptr);
  }
  inline reference operator[](difference_type const n) const noexcept {
    return int_range(m_ptr[n], m_ptr[n + 1]);
  }
  inline bool operator<(int_range_sum_iterator const& other) const noexcept {
    return m_ptr < other.m_ptr;
  }
  inline bool operator>(int_range_sum_iterator const& other) const noexcept {
    return m_ptr > other.m_ptr;
  }
  inline bool operator<=(int_range_sum_iterator const& other) const noexcept {
    return m_ptr <= other.m_ptr;
  }
  inline bool operator>=(int_range_sum_iterator const& other) const noexcept {
    return m_ptr >= other.m_ptr;
  }
};

template <class Allocator>
class int_range_sum {
  vector<int, Allocator> m_vector;
public:
  using value_type = int_range;
  using allocator_type = Allocator;
  using size_type = int;
  using difference_type = int;
  using reference = value_type;
  using const_reference = value_type;
  using pointer = value_type*;
  using const_pointer = value_type const*;
  using iterator = int_range_sum_iterator;
  using const_iterator = int_range_sum_iterator;
  explicit int_range_sum(Allocator const&);
  int_range_sum(int_range_sum&&) noexcept = default;
  int_range_sum(int_range_sum const&) = delete;
  int_range_sum& operator=(int_range_sum const&) = delete;
  int_range_sum& operator=(int_range_sum&&) = delete;
  ~int_range_sum() = default;
  const_iterator begin() const noexcept;
  const_iterator cbegin() const noexcept;
  const_iterator end() const noexcept;
  const_iterator cend() const noexcept;
  bool empty() const noexcept;
  size_type size() const noexcept;
  void clear();
  void swap(int_range_sum& other);
  void assign_sizes(vector<int, Allocator> const& sizes);
  void resize(size_type count);
};

}

#include <lgr_int_range_sum_inline.hpp>
