#pragma once

#include <iterator>
#include <numeric>
#include <lgr_inclusive_scan.hpp>

namespace lgr {

template <class Allocator>
int_range_sum<Allocator>::int_range_sum(Allocator const& allocator_in)
:m_vector(allocator_in)
{
}

template <class Allocator>
typename int_range_sum<Allocator>::iterator int_range_sum<Allocator>::begin() const noexcept {
  return iterator(m_vector.data());
}

template <class Allocator>
typename int_range_sum<Allocator>::iterator int_range_sum<Allocator>::cbegin() const noexcept {
  return iterator(m_vector.data());
}

template <class Allocator>
typename int_range_sum<Allocator>::iterator int_range_sum<Allocator>::end() const noexcept {
  return iterator(m_vector.data() + size());
}

template <class Allocator>
typename int_range_sum<Allocator>::iterator int_range_sum<Allocator>::cend() const noexcept {
  return iterator(m_vector.data() + size());
}

template <class Allocator>
bool int_range_sum<Allocator>::empty() const noexcept {
  return m_vector.size() <= 1;
}

template <class Allocator>
typename int_range_sum<Allocator>::size_type int_range_sum<Allocator>::size() const noexcept {
  return std::max(1, m_vector.size()) - 1;
}

template <class Allocator>
void int_range_sum<Allocator>::clear() {
  return m_vector.clear();
}

template <class Allocator>
void int_range_sum<Allocator>::swap(int_range_sum& other) {
  m_vector.swap(other.m_vector);
}

template <class Allocator>
void int_range_sum<Allocator>::assign_sizes(vector<int, Allocator> const& sizes) {
  assert(m_vector.size() == sizes.size() + 1);
  typename decltype(m_vector)::iterator offsets_iterator = m_vector.begin();
  *offsets_iterator = 0;
  ++offsets_iterator;
  lgr::inclusive_scan(sizes, iterator_range<decltype(offsets_iterator)>(offsets_iterator, m_vector.end()));
}

template <class Allocator>
void int_range_sum<Allocator>::resize(size_type count) {
  m_vector.resize(count + 1);
}

}
