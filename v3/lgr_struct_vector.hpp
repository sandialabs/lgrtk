#pragma once

#include <lgr_array_vector.hpp>
#include <lgr_struct_in_vector.hpp>

namespace lgr {

template <class T, layout L>
class struct_vector_iterator {
  using fundamental_type = typename struct_in_vector<T, L>::fundamental_type;
  outer_iterator<vector_iterator<fundamental_type>, L, int, int> m_array_iterator;
public:
  using value_type = std::remove_const_t<T>;
  using difference_type = int;
  using reference = struct_in_vector<T, L>;
  using iterator_category = std::random_access_iterator_tag;
  using pointer = void*;
  explicit inline struct_vector_iterator(
      decltype(m_array_iterator) const& array_iterator_in)
    :m_array_iterator(array_iterator_in)
  {
  }
  inline bool operator==(struct_vector_iterator const& other) const noexcept {
    return m_array_iterator == other.m_array_iterator;
  }
  inline bool operator!=(struct_vector_iterator const& other) const noexcept {
    return m_array_iterator != other.m_array_iterator;
  }
  inline reference operator*() const noexcept {
    return reference(*m_array_iterator);
  }
  inline struct_vector_iterator& operator++() noexcept {
    ++m_array_iterator;
    return *this;
  }
  inline struct_vector_iterator operator++(int) noexcept {
    auto ret = *this;
    ++m_array_iterator;
    return ret;
  }
  inline struct_vector_iterator& operator--() noexcept {
    --m_array_iterator;
    return *this;
  }
  inline struct_vector_iterator operator--(int) noexcept {
    auto ret = *this;
    --m_array_iterator;
    return ret;
  }
  inline struct_vector_iterator& operator+=(
      difference_type const n) noexcept {
    m_array_iterator += n;
    return *this;
  }
  inline struct_vector_iterator& operator-=(
      difference_type const n) noexcept {
    m_array_iterator -= n;
    return *this;
  }
  inline struct_vector_iterator operator+(
      difference_type const n) const noexcept {
    return struct_vector_iterator(
        m_array_iterator + n);
  }
  inline struct_vector_iterator operator-(
      difference_type const n) const noexcept {
    return struct_vector_iterator(
        m_array_iterator - n);
  }
  inline reference operator[](difference_type const n) const noexcept {
    return reference(m_array_iterator[n]);
  }
  inline bool operator<(struct_vector_iterator const& other) const noexcept {
    return m_array_iterator < other.m_array_iterator;
  }
  inline bool operator>(struct_vector_iterator const& other) const noexcept {
    return m_array_iterator > other.m_array_iterator;
  }
  inline bool operator<=(struct_vector_iterator const& other) const noexcept {
    return m_array_iterator <= other.m_array_iterator;
  }
  inline bool operator>=(struct_vector_iterator const& other) const noexcept {
    return m_array_iterator >= other.m_array_iterator;
  }
};

template <class T, layout L, class Allocator>
class struct_vector {
protected:
  using fundamental_type = typename struct_in_vector<T, L>::fundamental_type;
public:
  using allocator_type = typename Allocator::template rebind<fundamental_type>::other;
protected:
  static constexpr int array_size = struct_in_vector<T, L>::fundamental_array_size;
  array_vector<fundamental_type, array_size, L, allocator_type>
    m_array_vector;
public:
  using value_type = T;
  using size_type = int;
  using difference_type = int;
  using reference = struct_in_vector<T, L>;
  using const_reference = struct_in_vector<T const, L>;
  using iterator = struct_vector_iterator<T, L>;
  using const_iterator = struct_vector_iterator<T const, L>;
  explicit struct_vector(Allocator const& allocator_in) noexcept
    :m_array_vector(allocator_in)
  {}
  explicit struct_vector(size_type count, Allocator const& allocator_in)
    :m_array_vector(count, allocator_in)
  {}
  struct_vector(struct_vector&&) noexcept = default;
  struct_vector(struct_vector const&) = delete;
  struct_vector& operator=(struct_vector const&) = delete;
  struct_vector& operator=(struct_vector&&) = delete;
  ~struct_vector() = default;
  iterator begin() noexcept {
    return iterator(m_array_vector.begin());
  }
  const_iterator begin() const noexcept {
    return const_iterator(m_array_vector.begin());
  }
  const_iterator cbegin() const noexcept {
    return const_iterator(m_array_vector.cbegin());
  }
  iterator end() noexcept {
    return iterator(m_array_vector.end());
  }
  const_iterator end() const noexcept {
    return const_iterator(m_array_vector.end());
  }
  const_iterator cend() const noexcept {
    return const_iterator(m_array_vector.cend());
  }
  bool empty() const noexcept { return m_array_vector.empty(); }
  size_type size() const noexcept { return m_array_vector.size(); }
  void clear() { m_array_vector.clear(); }
  void resize(size_type count) { m_array_vector.resize(count); }
  void swap(struct_vector& other) { m_array_vector.swap(other.m_array_vector); }
  decltype(m_array_vector)& get_array_vector() {
    return m_array_vector;
  }
  decltype(m_array_vector) const& get_array_vector() const {
    return m_array_vector;
  }
};

}
