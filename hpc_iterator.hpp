#pragma once

#include <cassert>
#include <cstddef>
#include <iterator>
#include <hpc_macros.hpp>
#include <hpc_limits.hpp>

namespace hpc {

namespace impl {

template <class Iterator>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr typename std::iterator_traits<Iterator>::difference_type
distance(Iterator first, Iterator last, std::random_access_iterator_tag) noexcept {
  return last - first;
}

template <class Iterator>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr typename std::iterator_traits<Iterator>::difference_type
distance(Iterator first, Iterator last, std::input_iterator_tag) noexcept {
  using difference_type = typename std::iterator_traits<Iterator>::difference_type;
  difference_type i(0);
  for (; first != last; ++first) ++i;
  return i;
}

template <class Iterator>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr Iterator
advance(Iterator first, typename std::iterator_traits<Iterator>::difference_type n, std::random_access_iterator_tag) noexcept {
  return first + n;
}

template <class Iterator>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr Iterator
advance(Iterator first, typename std::iterator_traits<Iterator>::difference_type n, std::input_iterator_tag) noexcept {
  auto last = first;
  for (decltype(n) i(0); i != n; ++i) ++last;
  return last;
}

}

template <class Iterator>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr typename std::iterator_traits<Iterator>::difference_type
distance(Iterator first, Iterator last) noexcept {
  using tag_type = typename std::iterator_traits<Iterator>::iterator_category;
  return hpc::impl::distance(first, last, tag_type());
}

template <class Iterator>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr Iterator
advance(Iterator first,  typename std::iterator_traits<Iterator>::difference_type n) noexcept {
  using tag_type = typename std::iterator_traits<Iterator>::iterator_category;
  return hpc::impl::advance(first, n, tag_type());
}

template <class T, class Index = std::ptrdiff_t>
class pointer_iterator {
  T* m_pointer;
#ifndef NDEBUG
#error "need to be fast on GPUs right now"
  T* m_allocation_begin;
  T* m_allocation_end;
#endif
 public:
  using value_type = std::remove_const_t<T>;
  using difference_type = Index;
  using reference = T&;
  using pointer = T*;
  using iterator_category = std::random_access_iterator_tag;
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE explicit constexpr pointer_iterator(T* pointer_in) noexcept
    :m_pointer(pointer_in)
#ifndef NDEBUG
    ,m_allocation_begin(nullptr),m_allocation_end(pointer(nullptr) + ::hpc::numeric_limits<std::ptrdiff_t>::max())
#endif
  {}
#ifdef NDEBUG
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr pointer_iterator(T* pointer_in, T*, T*) noexcept
    :m_pointer(pointer_in)
  {}
#else
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr pointer_iterator(T* pointer_in, T* alloc_begin, T* alloc_end) noexcept
    :m_pointer(pointer_in)
    ,m_allocation_begin(alloc_begin),m_allocation_end(alloc_end)
  {}
#endif
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr bool operator==(pointer_iterator const& other) const noexcept {
    return m_pointer == other.m_pointer;
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr bool operator!=(pointer_iterator const& other) const noexcept {
    return m_pointer != other.m_pointer;
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr reference operator*() const noexcept {
#ifndef NDEBUG
    assert(m_allocation_begin <= m_pointer);
    assert(m_pointer < m_allocation_end);
#endif
    return *m_pointer;
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE pointer_iterator& operator++() noexcept {
    ++m_pointer;
    return *this;
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE pointer_iterator operator++(int) noexcept {
    auto ret = *this;
    ++m_pointer;
    return ret;
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE pointer_iterator& operator--() noexcept {
    --m_pointer;
    return *this;
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE pointer_iterator operator--(int) noexcept {
    auto ret = *this;
    --m_pointer;
    return ret;
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE pointer_iterator& operator+=(difference_type const n) noexcept {
    m_pointer += std::ptrdiff_t(n);
    return *this;
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE pointer_iterator& operator-=(difference_type const n) noexcept {
    m_pointer -= std::ptrdiff_t(n);
    return *this;
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr pointer_iterator operator+(difference_type const n) const noexcept {
    return pointer_iterator(m_pointer + std::ptrdiff_t(n)
#ifndef NDEBUG
        , m_allocation_begin, m_allocation_end
#endif
        );
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr pointer_iterator operator-(difference_type const n) const noexcept {
    return pointer_iterator(m_pointer - std::ptrdiff_t(n)
#ifndef NDEBUG
        , m_allocation_begin, m_allocation_end
#endif
        );
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr difference_type operator-(pointer_iterator const& other) const noexcept {
    return difference_type(m_pointer - other.m_pointer);
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr reference operator[](difference_type const i) const noexcept {
    return *((*this) + i);
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr bool operator<(pointer_iterator const& other) const noexcept {
    return m_pointer < other.m_pointer;
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr bool operator>(pointer_iterator const& other) const noexcept {
    return m_pointer > other.m_pointer;
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr bool operator<=(pointer_iterator const& other) const noexcept {
    return m_pointer <= other.m_pointer;
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr bool operator>=(pointer_iterator const& other) const noexcept {
    return m_pointer >= other.m_pointer;
  }
};

template <class T>
class counting_iterator {
  T m_value;
 public:
  using value_type = T;
  using difference_type = decltype(m_value - m_value);
  using reference = T;
  using pointer = T const*;
  using iterator_category = std::random_access_iterator_tag;
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr explicit counting_iterator(T value_in) noexcept : m_value(value_in) {}
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr bool operator==(counting_iterator const& other) const noexcept {
    return m_value == other.m_value;
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr bool operator!=(counting_iterator const& other) const noexcept {
    return m_value != other.m_value;
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr reference operator*() const noexcept { return m_value; }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE counting_iterator& operator++() noexcept {
    ++m_value;
    return *this;
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE counting_iterator operator++(int) noexcept {
    auto ret = *this;
    ++m_value;
    return ret;
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE counting_iterator& operator--() noexcept {
    --m_value;
    return *this;
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE counting_iterator operator--(int) noexcept {
    auto ret = *this;
    --m_value;
    return ret;
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE counting_iterator& operator+=(difference_type const n) noexcept {
    m_value += n;
    return *this;
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE counting_iterator& operator-=(difference_type const n) noexcept {
    m_value -= n;
    return *this;
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr counting_iterator operator+(difference_type const n) const noexcept {
    return counting_iterator(m_value + n);
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr counting_iterator operator-(difference_type const n) const noexcept {
    return counting_iterator(m_value - n);
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr difference_type operator-(counting_iterator const& other) const noexcept {
    return m_value - other.m_value;
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr reference operator[](difference_type const n) const noexcept {
    return *((*this) + n);
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr bool operator<(counting_iterator const& other) const noexcept {
    return m_value < other.m_value;
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr bool operator>(counting_iterator const& other) const noexcept {
    return m_value > other.m_value;
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr bool operator<=(counting_iterator const& other) const noexcept {
    return m_value <= other.m_value;
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr bool operator>=(counting_iterator const& other) const noexcept {
    return m_value >= other.m_value;
  }
};

}
