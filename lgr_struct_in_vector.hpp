#pragma once

#include <lgr_array_in_vector.hpp>
#include <lgr_vector3.hpp>

namespace lgr {

template <class T, layout L, class Index>
class struct_in_vector {
  array_in_vector<T, 1, L, Index> m_array;
public:
  using fundamental_type = T;
  static constexpr int fundamental_array_size = 1;
  inline struct_in_vector(decltype(m_array) const& array_in) noexcept
    :m_array(array_in)
  {}
  inline operator T() const noexcept {
    return m_array[0];
  }
  inline void operator=(T const& value) {
    m_array[0] = value;
  }
};

template <class T, layout L, class Index>
class struct_in_vector<T const, L, Index> {
  array_in_vector<T const, 1, L, Index> m_array;
public:
  using fundamental_type = T const;
  static constexpr int fundamental_array_size = 1;
  inline struct_in_vector(decltype(m_array) const& array_in) noexcept
    :m_array(array_in)
  {}
  inline operator T() const noexcept {
    return m_array[0];
  }
};

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

}
