#pragma once

#include <lgr_struct_vector.hpp>
#include <lgr_symmetric3x3.hpp>
#include <lgr_allocator.hpp>
#include <lgr_index.hpp>

namespace lgr {

template <class T, layout L, class Index>
class struct_in_vector<vector3<T>, L, Index> {
  array_in_vector<T, 3, L, Index> m_array;
public:
  using fundamental_type = T;
  static constexpr int fundamental_array_size = 3;
  inline struct_in_vector(decltype(m_array) const& array_in) noexcept
    :m_array(array_in)
  {}
  inline operator vector3<T>() const noexcept {
    T const a = m_array[0];
    T const b = m_array[1];
    T const c = m_array[2];
    return vector3<T>(a, b, c);
  }
  inline void operator=(vector3<T> const value) const noexcept {
    m_array[0] = value(0);
    m_array[1] = value(1);
    m_array[2] = value(2);
  }
};

template <class T, layout L, class Index>
class struct_in_vector<vector3<T> const, L, Index> {
  array_in_vector<T const, 3, L, Index> m_array;
public:
  using fundamental_type = T const;
  static constexpr int fundamental_array_size = 3;
  inline struct_in_vector(decltype(m_array) const& array_in) noexcept
    :m_array(array_in)
  {}
  inline operator vector3<T>() const noexcept {
    T const a = m_array[0];
    T const b = m_array[1];
    T const c = m_array[2];
    return vector3<T>(a, b, c);
  }
};

template <class T, layout L, class Index>
class struct_in_vector<matrix3x3<T>, L, Index> {
  array_in_vector<T, 9, L, Index> m_array;
public:
  using fundamental_type = T;
  static constexpr int fundamental_array_size = 9;
  inline struct_in_vector(decltype(m_array) const& array_in) noexcept
    :m_array(array_in)
  {}
  inline operator matrix3x3<T>() const noexcept {
    T const a = m_array[0];
    T const b = m_array[1];
    T const c = m_array[2];
    T const d = m_array[3];
    T const e = m_array[4];
    T const f = m_array[5];
    T const g = m_array[6];
    T const h = m_array[7];
    T const i = m_array[8];
    return matrix3x3<T>(a, b, c, d, e, f, g, h, i);
  }
  inline void operator=(matrix3x3<T> const value) const noexcept {
    m_array[0] = value(0, 0);
    m_array[1] = value(0, 1);
    m_array[2] = value(0, 2);
    m_array[3] = value(1, 0);
    m_array[4] = value(1, 1);
    m_array[5] = value(1, 2);
    m_array[6] = value(2, 0);
    m_array[7] = value(2, 1);
    m_array[8] = value(2, 2);
  }
};

template <class T, layout L, class Index>
class struct_in_vector<matrix3x3<T> const, L, Index> {
  array_in_vector<T const, 9, L, Index> m_array;
public:
  using fundamental_type = T const;
  static constexpr int fundamental_array_size = 9;
  inline struct_in_vector(decltype(m_array) const& array_in) noexcept
    :m_array(array_in)
  {}
  inline operator matrix3x3<T>() const noexcept {
    double const a = m_array[0];
    double const b = m_array[1];
    double const c = m_array[2];
    double const d = m_array[3];
    double const e = m_array[4];
    double const f = m_array[5];
    double const g = m_array[6];
    double const h = m_array[7];
    double const i = m_array[8];
    return matrix3x3<T>(a, b, c, d, e, f, g, h, i);
  }
};

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
  device_vector(device_vector&&) noexcept = default;
  device_vector(device_vector const&) = delete;
  device_vector& operator=(device_vector const&) = delete;
  device_vector& operator=(device_vector&&) = delete;
  ~device_vector() = default;
};

class node_index : public index<int, node_index> {
  public:
    using base_type = index<int, node_index>;
    constexpr explicit inline node_index(int const i) noexcept : base_type(i) {}
};

class node_in_element_index : public index<int, node_in_element_index> {
  public:
    using base_type = index<int, node_in_element_index>;
    constexpr explicit inline node_in_element_index(int const i) noexcept : base_type(i) {}
};

class element_index : public index<int, element_index> {
  public:
    using base_type = index<int, element_index>;
    constexpr explicit inline element_index(int const i) noexcept : base_type(i) {}
};

class element_node_index : public index<int, element_node_index> {
  public:
    using base_type = index<int, element_node_index>;
    constexpr explicit inline element_node_index(int const i) noexcept : base_type(i) {}
};

constexpr inline element_node_index operator*(element_index const& e, node_in_element_index const& n) noexcept {
  return element_node_index(int(e) * int(n));
}

constexpr inline element_index operator/(element_node_index const& en, node_in_element_index const& n) noexcept {
  return element_index(int(en) / int(n));
}

}
