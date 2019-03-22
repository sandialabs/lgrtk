#pragma once

#include <lgr_vector.hpp>
#include <lgr_array_in_vector.hpp>
#include <lgr_product_range.hpp>

namespace lgr {

template <class T, int N, layout L, class Allocator>
class array_vector {
  vector<T, Allocator> m_vector;
  product_range<vector_iterator<T>, L, int, int> array_range() noexcept {
    return product_range<vector_iterator<T>, L, int, int>(m_vector.begin(), size(), N);
  }
  product_range<vector_iterator<T const>, L, int, int> const_array_range() const noexcept {
    return product_range<vector_iterator<T const>, L, int, int>(m_vector.begin(), size(), N);
  }
public:
  using value_type = array_in_vector<T, N, L>;
  using allocator_type = Allocator;
  using size_type = int;
  using difference_type = int;
  using reference = array_in_vector<T, N, L>;
  using const_reference = array_in_vector<T const, N, L>;
  using iterator = outer_iterator<vector_iterator<T>, L, int, int>;
  using const_iterator = outer_iterator<vector_iterator<T const>, L, int, int>;
  explicit array_vector(Allocator const& allocator_in) noexcept
    :m_vector(allocator_in)
  {}
  explicit array_vector(size_type count, Allocator const& allocator_in)
    :m_vector(count * N, allocator_in)
  {}
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
    return m_vector.size() / N;
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
