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

constexpr inline element_node operator*(node_in_element const& n, element const& e) noexcept {
  return e * n;
}

}
