#pragma once

#include <lgr_struct_vector.hpp>
#include <lgr_allocator.hpp>
#include <lgr_index.hpp>

namespace lgr {

template <class T>
class host_vector : public struct_vector<T, AOS, host_allocator<T>> {
  using base_type = struct_vector<T, AOS, host_allocator<T>>;
public:
  using typename base_type::size_type;
  explicit host_vector() noexcept
    :base_type(host_allocator<T>())
  {}
  explicit host_vector(size_type count)
    :base_type(count, host_allocator<T>())
  {}
  host_vector(host_vector&&) noexcept = default;
  host_vector(host_vector const&) = delete;
  host_vector& operator=(host_vector const&) = delete;
  host_vector& operator=(host_vector&&) = delete;
  ~host_vector() = default;
};

class node_index : public index<int, node_index> {
  public:
    using base_type = index<int, node_index>;
    constexpr explicit inline node_index(int const i) : base_type(i) {}
};

}
