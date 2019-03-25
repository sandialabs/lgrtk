#pragma once

#include <lgr_vector.hpp>
#include <lgr_array_in_vector.hpp>
#include <lgr_product_range.hpp>

namespace lgr {

template <class T, int N, layout L, class Allocator, class OuterIndex = int, class InnerIndex = int>
class array_vector {
  using product_index = decltype(std::declval<OuterIndex>() * std::declval<InnerIndex>());
  vector<T, Allocator, product_index> m_vector;
  using product_iterator = typename decltype(m_vector)::iterator;
  using const_product_iterator = typename decltype(m_vector)::const_iterator;
  using range_type = product_range<product_iterator, L, OuterIndex, InnerIndex>;
  using const_range_type = product_range<const_product_iterator, L, OuterIndex, InnerIndex>;
  range_type array_range() noexcept {
    return range_type(m_vector.begin(), size(), InnerIndex(N));
  }
  const_range_type const_array_range() const noexcept {
    return const_range_type(m_vector.begin(), size(), InnerIndex(N));
  }
public:
  using value_type = array_in_vector<T, N, L, OuterIndex, InnerIndex>;
  using allocator_type = Allocator;
  using size_type = OuterIndex;
  using difference_type = OuterIndex;
  using reference = array_in_vector<T, N, L, OuterIndex, InnerIndex>;
  using const_reference = array_in_vector<T const, N, L, OuterIndex, InnerIndex>;
  using iterator = typename range_type::iterator;
  using const_iterator = typename const_range_type::iterator;
  explicit array_vector(Allocator const& allocator_in) noexcept
    :m_vector(allocator_in)
  {}
  explicit array_vector(size_type count, Allocator const& allocator_in)
    :m_vector(count * InnerIndex(N), allocator_in)
  {
    if (m_vector.size() != count * InnerIndex(N)) {
      m_vector.clear();
    }
  }
  array_vector(array_vector&&) noexcept = default;
  array_vector(array_vector const&) = delete;
  array_vector& operator=(array_vector const&) = delete;
  array_vector& operator=(array_vector&&) = delete;
  ~array_vector() = default;
  iterator begin() noexcept {
    return array_range().begin();
  }
  const_iterator begin() const noexcept {
    return const_array_range().begin();
  }
  const_iterator cbegin() const noexcept {
    return const_array_range().begin();
  }
  iterator end() noexcept {
    return array_range().end();
  }
  const_iterator end() const noexcept {
    return const_array_range().end();
  }
  const_iterator cend() const noexcept {
    return const_array_range().end();
  }
  bool empty() const noexcept {
    return m_vector.empty();
  }
  size_type size() const noexcept {
    return m_vector.size() / InnerIndex(N);
  }
  void clear() {
    m_vector.clear();
  }
  void resize(size_type count) {
    m_vector.resize(count * N);
  }
  void swap(array_vector& other) {
    m_vector.swap(other);
  }
  allocator_type get_allocator() const {
    return m_vector.get_allocator();
  }
  decltype(m_vector)& get_vector() {
    return m_vector;
  }
  decltype(m_vector) const& get_vector() const {
    return m_vector;
  }
};

}
