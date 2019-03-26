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

static constexpr layout device_layout = AOS;

template <class T, class Index>
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
  device_vector(device_vector&&) noexcept = default;
  device_vector(device_vector const&) = delete;
  device_vector& operator=(device_vector const&) = delete;
  device_vector& operator=(device_vector&&) = delete;
  ~device_vector() = default;
};

class node : public index<int, node> {
  public:
    using base_type = index<int, node>;
    constexpr explicit inline node(int const i) noexcept : base_type(i) {}
};

class node_in_element : public index<int, node_in_element> {
  public:
    using base_type = index<int, node_in_element>;
    constexpr explicit inline node_in_element(int const i) noexcept : base_type(i) {}
};

class element : public index<int, element> {
  public:
    using base_type = index<int, element>;
    constexpr explicit inline element(int const i) noexcept : base_type(i) {}
};

class element_node : public index<int, element_node> {
  public:
    using base_type = index<int, element_node>;
    constexpr explicit inline element_node(int const i) noexcept : base_type(i) {}
};

constexpr inline element_node operator*(element const& e, node_in_element const& n) noexcept {
  return element_node(int(e) * int(n));
}

constexpr inline element operator/(element_node const& en, node_in_element const& n) noexcept {
  return element(int(en) / int(n));
}

}
