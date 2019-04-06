#pragma once

#include <lgr_struct_vector.hpp>

namespace lgr {

template <class T, class Index = int>
class pinned_vector : public struct_vector<T, device_layout, pinned_allocator<T>, Index> {
  using base_type = struct_vector<T, device_layout, pinned_allocator<T>, Index>;
public:
  using typename base_type::size_type;
  explicit pinned_vector(pinned_memory_pool& pool_in) noexcept
    :base_type(pinned_allocator<T>(pool_in))
  {}
  explicit pinned_vector(size_type count, pinned_memory_pool& pool_in)
    :base_type(count, pinned_allocator<T>(pool_in))
  {}
  pinned_vector(pinned_vector&&) noexcept = default;
  pinned_vector(pinned_vector const&) = delete;
  pinned_vector& operator=(pinned_vector const&) = delete;
  pinned_vector& operator=(pinned_vector&&) = delete;
  ~pinned_vector() = default;
  typename base_type::reference operator[](Index const i) { return this->begin()[i]; }
  typename base_type::const_reference operator[](Index const i) const { return this->cbegin()[i]; }
};

}
