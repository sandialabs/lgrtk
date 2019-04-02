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
  explicit inline pointer_iterator(T* ptr_in) noexcept : m_ptr(ptr_in) {}
  inline bool operator==(pointer_iterator const& other) const noexcept {
    return m_ptr == other.m_ptr;
  }
  inline bool operator!=(pointer_iterator const& other) const noexcept {
    return m_ptr != other.m_ptr;
  }
  inline reference operator*() const noexcept { return *m_ptr; }
  inline pointer_iterator& operator++() noexcept {
    ++m_ptr;
    return *this;
  }
  inline pointer_iterator operator++(int) noexcept {
    auto ret = *this;
    ++m_ptr;
    return ret;
  }
  inline pointer_iterator& operator--() noexcept {
    --m_ptr;
    return *this;
  }
  inline pointer_iterator operator--(int) noexcept {
    auto ret = *this;
    --m_ptr;
    return ret;
  }
  inline pointer_iterator& operator+=(difference_type const n) noexcept {
    m_ptr += int(n);
    return *this;
  }
  inline pointer_iterator& operator-=(difference_type const n) noexcept {
    m_ptr -= int(n);
    return *this;
  }
  inline pointer_iterator operator+(difference_type const n) const noexcept {
    return pointer_iterator(m_ptr + int(n));
  }
  inline pointer_iterator operator-(difference_type const n) const noexcept {
    return pointer_iterator(m_ptr - int(n));
  }
  inline difference_type operator-(pointer_iterator const& other) const noexcept {
    return difference_type(m_ptr - other.m_ptr);
  }
  inline reference operator[](Index const i) const noexcept {
    return *(m_ptr + int(i));
  }
  inline bool operator<(pointer_iterator const& other) const noexcept {
    return m_ptr < other.m_ptr;
  }
  inline bool operator>(pointer_iterator const& other) const noexcept {
    return m_ptr > other.m_ptr;
  }
  inline bool operator<=(pointer_iterator const& other) const noexcept {
    return m_ptr <= other.m_ptr;
  }
  inline bool operator>=(pointer_iterator const& other) const noexcept {
    return m_ptr >= other.m_ptr;
  }
};

}
