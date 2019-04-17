#pragma once

#include <lgr_struct_vector.hpp>
#include <lgr_allocator.hpp>

namespace lgr {

template <class T, class Index = int>
class device_vector : public struct_vector<T, device_layout, device_allocator<T>, Index> {
  using base_type = struct_vector<T, device_layout, device_allocator<T>, Index>;
public:
  using typename base_type::size_type;
  explicit device_vector(device_memory_pool& pool_in) noexcept
    :base_type(device_allocator<T>(pool_in))
  {}
  explicit device_vector(size_type count, device_memory_pool& pool_in)
    :base_type(count, device_allocator<T>(pool_in))
  {}
  explicit device_vector(size_type count, device_allocator<T> const& alloc_in)
    :base_type(count, alloc_in)
  {}
  device_vector(device_vector&&) noexcept = default;
  device_vector(device_vector const&) = delete;
  device_vector& operator=(device_vector const&) = delete;
  device_vector& operator=(device_vector&&) noexcept = default;
  ~device_vector() = default;
};

}
