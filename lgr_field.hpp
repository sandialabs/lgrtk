#pragma once

#include <lgr_struct_vector.hpp>
#include <lgr_allocator.hpp>

namespace lgr {

template <class T, layout L, class Allocator>
class field {
protected:
  using vector_type = vector<struct_in_vector<T, L>, Allocator>;
  vector_type m_vector;
  int_range const* m_range;
public:
  using value_type = typename vector_type::value_type;
  using allocator_type = typename vector_type::allocator_type;
  using size_type = typename vector_type::size_type;
  using difference_type = typename vector_type::difference_type;
  using reference = typename vector_type::reference;
  using const_reference = typename vector_type::const_reference;
  using iterator = typename vector_type::iterator;
  using const_iterator = typename vector_type::const_iterator;
  explicit field(int_range const& range_in, Allocator const& allocator_in) noexcept
    :m_vector(allocator_in)
    ,m_range(&range_in)
  {}
  field(field&&) noexcept = default;
  field(field const&) = delete;
  field& operator=(field const&) = delete;
  field& operator=(field&&) = delete;
  ~field() = default;
  iterator begin() noexcept {
    assert(size() == m_range->size());
    return m_vector.begin();
  }
  const_iterator begin() const noexcept {
    assert(size() == m_range->size());
    return m_vector.begin();
  }
  const_iterator cbegin() const noexcept {
    assert(size() == m_range->size());
    return m_vector.cbegin();
  }
  iterator end() noexcept { return m_vector.end(); }
  const_iterator end() const noexcept { return m_vector.end(); }
  const_iterator cend() const noexcept { return m_vector.cend(); }
  bool empty() const noexcept { return m_vector.empty(); }
  size_type size() const noexcept { return m_vector.size(); }
  void clear() { m_vector.clear(); }
  int_range range() const noexcept {
    return *m_range;
  }
  void resize() { m_vector.resize(m_range->size()); }
  void swap(field& other) { m_vector.swap(other.m_vector); }
};

}
