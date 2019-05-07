#pragma once

namespace lgr {

template <class T>
class iterator_range {
  T m_begin;
  T m_end;
  public:
  using difference_type = typename T::difference_type;
  using reference = typename T::reference;
  using iterator = T;
  inline iterator_range(T const& begin_in, T const& end_in) noexcept
    :m_begin(begin_in)
    ,m_end(end_in)
  {
  }
  inline T begin() const noexcept {
    return m_begin;
  }
  inline T end() const noexcept {
    return m_end;
  }
  inline difference_type size() const noexcept {
    return m_end - m_begin;
  }
  inline reference operator[](difference_type const n) const noexcept {
    return m_begin[n];
  }
  inline void resize(difference_type const n) {
    m_end = m_begin + n;
  }
};

}

