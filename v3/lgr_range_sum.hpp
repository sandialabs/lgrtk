#pragma once

#include <cassert>

#include <lgr_counting_range.hpp>
#include <lgr_vector.hpp>
#include <lgr_inclusive_scan.hpp>
#include <lgr_binary_ops.hpp>

namespace lgr {

template <class SourceIndex, class TargetIndex>
class range_sum_iterator {
  TargetIndex const* m_ptr;

 public:
  using value_type = counting_range<TargetIndex>;
  using difference_type = SourceIndex;
  using reference = value_type;
  using pointer = value_type const*;
  using iterator_category = std::random_access_iterator_tag;
  explicit inline range_sum_iterator(TargetIndex const* ptr_in) noexcept : m_ptr(ptr_in) {}
  inline bool operator==(range_sum_iterator const& other) const noexcept {
    return m_ptr == other.m_ptr;
  }
  inline bool operator!=(range_sum_iterator const& other) const noexcept {
    return m_ptr != other.m_ptr;
  }
  inline reference operator*() const noexcept { return value_type(m_ptr[0], m_ptr[1]); }
  inline range_sum_iterator& operator++() noexcept {
    ++m_ptr;
    return *this;
  }
  inline range_sum_iterator operator++(int) noexcept {
    auto ret = *this;
    ++m_ptr;
    return ret;
  }
  inline range_sum_iterator& operator--() noexcept {
    --m_ptr;
    return *this;
  }
  inline range_sum_iterator operator--(int) noexcept {
    auto ret = *this;
    --m_ptr;
    return ret;
  }
  inline range_sum_iterator& operator+=(difference_type const n) noexcept {
    m_ptr += int(n);
    return *this;
  }
  inline range_sum_iterator& operator-=(difference_type const n) noexcept {
    m_ptr -= int(n);
    return *this;
  }
  inline range_sum_iterator operator+(difference_type const n) const noexcept {
    return range_sum_iterator(m_ptr + int(n));
  }
  inline range_sum_iterator operator-(difference_type const n) const noexcept {
    return range_sum_iterator(m_ptr - int(n));
  }
  inline difference_type operator-(range_sum_iterator const& other) const noexcept {
    return difference_type(int(m_ptr - other.m_ptr));
  }
  inline reference operator[](difference_type const n) const noexcept {
    return value_type(m_ptr[int(n)], m_ptr[int(n) + 1]);
  }
  inline bool operator<(range_sum_iterator const& other) const noexcept {
    return m_ptr < other.m_ptr;
  }
  inline bool operator>(range_sum_iterator const& other) const noexcept {
    return m_ptr > other.m_ptr;
  }
  inline bool operator<=(range_sum_iterator const& other) const noexcept {
    return m_ptr <= other.m_ptr;
  }
  inline bool operator>=(range_sum_iterator const& other) const noexcept {
    return m_ptr >= other.m_ptr;
  }
};

template <class TargetIndex, class Allocator, class SourceIndex>
class range_sum {
  vector<TargetIndex, Allocator, SourceIndex> m_vector;
public:
  using value_type = counting_range<TargetIndex>;
  using allocator_type = Allocator;
  using size_type = SourceIndex;
  using difference_type = SourceIndex;
  using reference = value_type;
  using const_reference = value_type;
  using pointer = value_type*;
  using const_pointer = value_type const*;
  using iterator = range_sum_iterator<SourceIndex, TargetIndex>;
  using const_iterator = range_sum_iterator<SourceIndex, TargetIndex>;
  explicit range_sum(Allocator const& allocator_in)
  :m_vector(allocator_in)
  {
  }
  range_sum(range_sum&&) noexcept = default;
  range_sum(range_sum const&) = delete;
  range_sum& operator=(range_sum const&) = delete;
  range_sum& operator=(range_sum&&) = delete;
  ~range_sum() = default;
  const_iterator begin() const noexcept {
    return iterator(m_vector.data());
  }
  const_iterator cbegin() const noexcept {
    return iterator(m_vector.data());
  }
  const_iterator end() const noexcept {
    return iterator(m_vector.data() + size());
  }
  const_iterator cend() const noexcept {
    return iterator(m_vector.data() + size());
  }
  bool empty() const noexcept {
    return m_vector.size() <= 1;
  }
  size_type size() const noexcept {
    return lgr::max(1, m_vector.size()) - 1;
  }
  void clear() {
    return m_vector.clear();
  }
  void swap(range_sum& other) {
    m_vector.swap(other.m_vector);
  }
  void assign_sizes(vector<int, typename Allocator::template rebind<int>::other, SourceIndex> const& sizes) {
    assert(m_vector.size() == sizes.size() + 1);
    typename decltype(m_vector)::iterator offsets_iterator = m_vector.begin();
    *offsets_iterator = 0;
    ++offsets_iterator;
    lgr::inclusive_scan(sizes, iterator_range<decltype(offsets_iterator)>(offsets_iterator, m_vector.end()));
  }
  void resize(size_type count) {
    m_vector.resize(count + size_type(1));
  }
};

}
