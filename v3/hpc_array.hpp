#pragma once

#include <initializer_list>

#include <hpc_iterator.hpp>
#include <hpc_array_traits.hpp>

namespace hpc {

template <
  class T,
  std::ptrdiff_t N,
  class Index = std::ptrdiff_t>
class array {
protected:
  T m_data[N];
public:
  using value_type = T;
  using size_type = Index;
  using difference_type = decltype(std::declval<Index>() - std::declval<Index>());
  using reference = value_type&;
  using const_reference = value_type const&;
  using pointer = T*;
  using const_pointer = T const*;
  using iterator = pointer_iterator<T, size_type>;
  using const_iterator = pointer_iterator<T const, size_type>;
  array(std::initializer_list<T> l) noexcept {
    auto const b = l.begin();
    for (std::ptrdiff_t i = 0; i < N; ++i) {
      new (m_data + i) T(b[i]);
    }
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE array() noexcept = default;
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE array(array&& other) noexcept = default;
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE array(array const&) noexcept = default;
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE array& operator=(array const&) noexcept = default;
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE array& operator=(array&& other) noexcept = default;
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr pointer data() noexcept { return m_data; }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr const_pointer data() const noexcept { return m_data; }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr iterator begin() noexcept {
    return iterator(m_data, m_data, m_data + N);
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr const_iterator begin() const noexcept {
    return const_iterator(m_data, m_data, m_data + N);
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr const_iterator cbegin() const noexcept {
    return const_iterator(m_data, m_data, m_data + N);
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr iterator end() noexcept {
    return iterator(m_data + N, m_data, m_data + N);
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr const_iterator end() const noexcept {
    return const_iterator(m_data + N, m_data, m_data + N);
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr const_iterator cend() const noexcept {
    return const_iterator(m_data + N, m_data, m_data + N);
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr bool empty() const noexcept {
    return N == 0;
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr size_type size() const noexcept { return size_type(N); }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr reference operator[](size_type i) noexcept { return begin()[i]; }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr const_reference operator[](size_type i) const noexcept { return begin()[i]; }
};

template <class T, std::ptrdiff_t N, class I>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr bool operator==(array<T, N, I> const& a, array<T, N, I> const& b) {
  for (I i(0); i != a.size(); ++i) {
    if (!(a[i] == b[i])) return false;
  }
  return true;
}

template <class T, std::ptrdiff_t N, class I>
class array_traits<::hpc::array<T, N, I>> {
  public:
  using value_type = T;
  HPC_HOST_DEVICE static constexpr std::ptrdiff_t size() noexcept { return N; }
  template <class Iterator>
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE static ::hpc::array<T, N, I> load(Iterator it) noexcept {
    ::hpc::array<T, N, I> result;
    auto it2 = result.begin();
    for (std::ptrdiff_t i = 0; i < N; ++i, ++it, ++it2) {
      *it2 = *it;
    }
    return result;
  }
  template <class Iterator>
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE static void store(Iterator it, ::hpc::array<T, N, I> const& value) noexcept {
    auto it2 = value.begin();
    for (std::ptrdiff_t i = 0; i < N; ++i, ++it, ++it2) {
      *it = *it2;
    }
  }
};

}
