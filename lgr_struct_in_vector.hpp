#pragma once

#include <lgr_array_in_vector.hpp>

namespace lgr {

template <class T, layout L>
class struct_in_vector {
  array_in_vector<T, 1, L> m_array;
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

template <class T, layout L>
class struct_in_vector<T const, L> {
  array_in_vector<T const, 1, L> m_array;
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

}
