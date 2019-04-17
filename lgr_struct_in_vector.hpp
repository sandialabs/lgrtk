#pragma once

#include <lgr_array_in_vector.hpp>
#include <lgr_vector3.hpp>
#include <lgr_matrix3x3.hpp>
#include <lgr_symmetric3x3.hpp>

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

template <class T, layout L, class Index>
class struct_in_vector<symmetric3x3<T>, L, Index> {
  array_in_vector<T, 6, L, Index> m_array;
public:
  using fundamental_type = T;
  static constexpr int fundamental_array_size = 6;
  inline struct_in_vector(decltype(m_array) const& array_in) noexcept
    :m_array(array_in)
  {}
  inline operator symmetric3x3<T>() const noexcept {
    T const a = m_array[0];
    T const b = m_array[1];
    T const c = m_array[2];
    T const d = m_array[3];
    T const e = m_array[4];
    T const f = m_array[5];
    return symmetric3x3<T>(a, b, c, d, e, f);
  }
  inline void operator=(symmetric3x3<T> const value) const noexcept {
    m_array[0] = value(symmetric3x3<T>::XX);
    m_array[1] = value(symmetric3x3<T>::YY);
    m_array[2] = value(symmetric3x3<T>::ZZ);
    m_array[3] = value(symmetric3x3<T>::XY);
    m_array[4] = value(symmetric3x3<T>::YZ);
    m_array[5] = value(symmetric3x3<T>::XZ);
  }
};

template <class T, layout L, class Index>
class struct_in_vector<symmetric3x3<T> const, L, Index> {
  array_in_vector<T const, 6, L, Index> m_array;
public:
  using fundamental_type = T const;
  static constexpr int fundamental_array_size = 6;
  inline struct_in_vector(decltype(m_array) const& array_in) noexcept
    :m_array(array_in)
  {}
  inline operator symmetric3x3<T>() const noexcept {
    T const a = m_array[0];
    T const b = m_array[1];
    T const c = m_array[2];
    T const d = m_array[3];
    T const e = m_array[4];
    T const f = m_array[5];
    return symmetric3x3<T>(a, b, c, d, e, f);
  }
};

}
