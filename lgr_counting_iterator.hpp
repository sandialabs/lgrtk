#pragma once

#include <iterator>

namespace lgr {

template <class T>
class counting_iterator {
  T i;

 public:
  using value_type = T;
  using difference_type = decltype(i - i);
  using reference = T;
  using pointer = T const*;
  using iterator_category = std::random_access_iterator_tag;
  inline counting_iterator() noexcept = default;
  explicit inline counting_iterator(T const i_in) noexcept : i(i_in) {}
  inline bool operator==(counting_iterator const& other) const noexcept {
    return i == other.i;
  }
  inline bool operator!=(counting_iterator const& other) const noexcept {
    return i != other.i;
  }
  inline reference operator*() const noexcept { return i; }
  inline counting_iterator& operator++() noexcept {
    ++i;
    return *this;
  }
  inline counting_iterator operator++(int) noexcept {
    auto ret = *this;
    ++i;
    return ret;
  }
  inline counting_iterator& operator--() noexcept {
    --i;
    return *this;
  }
  inline counting_iterator operator--(int) noexcept {
    auto ret = *this;
    --i;
    return ret;
  }
  inline counting_iterator& operator+=(difference_type const n) noexcept {
    i += n;
    return *this;
  }
  inline counting_iterator& operator-=(difference_type const n) noexcept {
    i -= n;
    return *this;
  }
  inline counting_iterator operator+(difference_type const n) const noexcept {
    return counting_iterator(i + n);
  }
  inline counting_iterator operator-(difference_type const n) const noexcept {
    return counting_iterator(i - n);
  }
  inline difference_type operator-(counting_iterator const& other) const
      noexcept {
    return i - other.i;
  }
  inline reference operator[](difference_type const n) const noexcept {
    return i + n;
  }
  inline bool operator<(counting_iterator const& other) const noexcept {
    return i < other.i;
  }
  inline bool operator>(counting_iterator const& other) const noexcept {
    return i > other.i;
  }
  inline bool operator<=(counting_iterator const& other) const noexcept {
    return i <= other.i;
  }
  inline bool operator>=(counting_iterator const& other) const noexcept {
    return i >= other.i;
  }
};

template <class T>
inline counting_iterator<T> operator+(
    typename counting_iterator<T>::difference_type const n, counting_iterator<T> const it) noexcept {
  return it + n;
}

}

