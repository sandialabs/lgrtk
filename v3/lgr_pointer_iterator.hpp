#pragma once

namespace lgr {

template <class T, class Index = int>
class pointer_iterator {
  T* m_ptr;
 public:
  using value_type = std::remove_const_t<T>;
  using difference_type = decltype(Index(0) - Index(0));
  using reference = T&;
  using pointer = T*;
  using iterator_category = std::random_access_iterator_tag;
  explicit inline vector_iterator(T* ptr_in) noexcept : m_ptr(ptr_in) {}
  inline bool operator==(vector_iterator const& other) const noexcept {
    return m_ptr == other.m_ptr;
  }
  inline bool operator!=(vector_iterator const& other) const noexcept {
    return m_ptr != other.m_ptr;
  }
  inline reference operator*() const noexcept { return *m_ptr; }
  inline vector_iterator& operator++() noexcept {
    ++m_ptr;
    return *this;
  }
  inline vector_iterator operator++(int) noexcept {
    auto ret = *this;
    ++m_ptr;
    return ret;
  }
  inline vector_iterator& operator--() noexcept {
    --m_ptr;
    return *this;
  }
  inline vector_iterator operator--(int) noexcept {
    auto ret = *this;
    --m_ptr;
    return ret;
  }
  inline vector_iterator& operator+=(difference_type const n) noexcept {
    m_ptr += int(n);
    return *this;
  }
  inline vector_iterator& operator-=(difference_type const n) noexcept {
    m_ptr -= int(n);
    return *this;
  }
  inline vector_iterator operator+(difference_type const n) const noexcept {
    return vector_iterator(m_ptr + int(n));
  }
  inline vector_iterator operator-(difference_type const n) const noexcept {
    return vector_iterator(m_ptr - int(n));
  }
  inline difference_type operator-(vector_iterator const& other) const noexcept {
    return difference_type(m_ptr - other.m_ptr);
  }
  inline reference operator[](Index const i) const noexcept {
    return *(m_ptr + int(i));
  }
  inline bool operator<(vector_iterator const& other) const noexcept {
    return m_ptr < other.m_ptr;
  }
  inline bool operator>(vector_iterator const& other) const noexcept {
    return m_ptr > other.m_ptr;
  }
  inline bool operator<=(vector_iterator const& other) const noexcept {
    return m_ptr <= other.m_ptr;
  }
  inline bool operator>=(vector_iterator const& other) const noexcept {
    return m_ptr >= other.m_ptr;
  }
};

}
