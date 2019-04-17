#pragma once

#include <lgr_pointer_iterator.hpp>

namespace lgr {

template <class T, int N, class Index = int>
class array {
  public:
    using value_type = T;
    using size_type = Index;
    using difference_type = Index;
    using reference = value_type&;
    using const_reference = value_type const&;
    using pointer = value_type*;
    using const_pointer = value_type const*;
    using iterator = pointer_iterator<T, Index>;
    using const_iterator = pointer_iterator<T const, Index>;
  private:
    T m_impl[N];
  public:
    inline reference operator[](Index const i) noexcept {
      return m_impl[int(i)];
    }
    inline const_reference operator[](Index const i) const noexcept {
      return m_impl[int(i)];
    }
    inline pointer data() noexcept { return m_impl; }
    inline const_pointer data() const noexcept { return m_impl; }
    inline iterator begin() noexcept { return iterator(m_impl); } 
    inline const_iterator begin() const noexcept { return const_iterator(m_impl); } 
    inline const_iterator cbegin() const noexcept { return const_iterator(m_impl); } 
    inline iterator end() noexcept { return iterator(m_impl + N); } 
    inline const_iterator end() const noexcept { return const_iterator(m_impl + N); } 
    inline const_iterator cend() const noexcept { return const_iterator(m_impl + N); } 
    constexpr size_type size() const noexcept { return size_type(N); }
    inline array& operator=(array const&) noexcept = default;
    inline array(array const&) noexcept = default;
    inline array() noexcept = default;
};

}
