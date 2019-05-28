#pragma once

#include <hpc_iterator.hpp>

namespace hpc {

namespace impl {

template <class Iterator, class Tag>
class iterator_range;

template <class Iterator>
class iterator_range<Iterator, std::input_iterator_tag> {
  Iterator m_begin;
  Iterator m_end;
 public:
  using difference_type = typename std::iterator_traits<Iterator>::difference_type;
  using reference = typename std::iterator_traits<Iterator>::reference;
  using value_type = typename std::iterator_traits<Iterator>::value_type;
  using iterator = Iterator;
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr iterator_range(Iterator begin_in, Iterator end_in) noexcept : m_begin(begin_in), m_end(end_in) {}
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr iterator begin() const noexcept { return m_begin; }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr iterator end() const noexcept { return m_end; }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE void resize(difference_type n) noexcept { m_end = ::hpc::advance(m_begin, n); }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr difference_type size() const noexcept { return ::hpc::distance(m_begin, m_end); }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr bool empty() const noexcept { return m_begin == m_end; }
};

template <class Iterator>
class iterator_range<Iterator, std::random_access_iterator_tag> : public ::hpc::impl::iterator_range<Iterator, std::input_iterator_tag> {
  using base_type = ::hpc::impl::iterator_range<Iterator, std::input_iterator_tag>;
 public:
  using difference_type = typename base_type::difference_type;
  using reference = typename base_type::reference;
  using value_type = typename base_type::value_type;
  using iterator = typename base_type::iterator;
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr iterator_range(iterator begin_in, iterator end_in) noexcept : base_type(begin_in, end_in) {}
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE reference operator[](difference_type n) const noexcept { return base_type::begin()[n]; }
};

}

template <class Iterator>
class iterator_range : public ::hpc::impl::iterator_range<Iterator, typename std::iterator_traits<Iterator>::iterator_category> {
  using base_type = ::hpc::impl::iterator_range<Iterator, typename std::iterator_traits<Iterator>::iterator_category>;
 public:
  using difference_type = typename base_type::difference_type;
  using reference = typename base_type::reference;
  using value_type = typename base_type::value_type;
  using iterator = typename base_type::iterator;
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr iterator_range(iterator begin_in, iterator end_in) noexcept : base_type(begin_in, end_in) {}
};

template <class Iterator>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr iterator_range<Iterator>
make_iterator_range(Iterator begin, Iterator end) noexcept {
  return iterator_range<Iterator>(begin, end);
}

template <class T>
class counting_range : public iterator_range<counting_iterator<T>> {
  using base_type = iterator_range<counting_iterator<T>>;
 public:
  using difference_type = typename base_type::difference_type;
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr explicit counting_range(T first, T last):
  iterator_range<counting_iterator<T>>(counting_iterator<T>(first), counting_iterator<T>(last))
  {
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr explicit counting_range(difference_type size_in):
  iterator_range<counting_iterator<T>>(counting_iterator<T>(T(0)), counting_iterator<T>(T(0) + size_in))
  {
  }
};

template <class T>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr counting_range<T>
make_counting_range(T begin, T end) noexcept {
  return counting_range<T>(begin, end);
}

template <class T>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr counting_range<T>
make_counting_range(T size) noexcept {
  return counting_range<T>(size);
}

enum class layout {
  left,
  right,
};

namespace impl {

template <class Iterator, layout L, class OuterIndex, class InnerIndex>
class inner_iterator;

template <class Iterator, layout L, class OuterIndex, class InnerIndex>
class outer_iterator;

template <class Iterator, class OuterIndex, class InnerIndex>
class inner_iterator<Iterator, layout::right, OuterIndex, InnerIndex> {
  Iterator m_begin;
  public:
  using value_type = typename Iterator::value_type;
  using size_type = InnerIndex;
  using difference_type = InnerIndex;
  using reference = typename Iterator::reference;
  using pointer = typename Iterator::pointer;
  using iterator_category = std::random_access_iterator_tag;
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr explicit inner_iterator(Iterator impl_in) noexcept
    :m_begin(impl_in)
  {
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr bool operator==(inner_iterator const& other) const noexcept {
    return m_begin == other.m_begin;
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr bool operator!=(inner_iterator const& other) const noexcept {
    return m_begin != other.m_begin;
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr reference operator*() const noexcept {
    return *m_begin;
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE inner_iterator& operator++() noexcept {
    ++m_begin;
    return *this;
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE inner_iterator operator++(int) noexcept {
    auto ret = *this;
    ++m_begin;
    return ret;
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE inner_iterator& operator--() noexcept {
    --m_begin;
    return *this;
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE inner_iterator operator--(int) noexcept {
    auto ret = *this;
    --m_begin;
    return ret;
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE inner_iterator& operator+=(difference_type const n) noexcept {
    m_begin += OuterIndex(1) * n;
    return *this;
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE inner_iterator& operator-=(difference_type const n) noexcept {
    m_begin -= OuterIndex(1) * n;
    return *this;
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr inner_iterator operator+(difference_type const n) const noexcept {
    return inner_iterator(m_begin + (OuterIndex(1) * n));
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr inner_iterator operator-(difference_type const n) const noexcept {
    return inner_iterator(m_begin - (OuterIndex(1) * n));
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr difference_type operator-(inner_iterator const other) const noexcept {
    return difference_type(std::ptrdiff_t(m_begin - other.m_begin));
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr reference operator[](difference_type const n) const noexcept {
    return *((*this) + n);
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr bool operator<(inner_iterator const& other) const noexcept {
    return m_begin < other.m_begin;
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr bool operator>(inner_iterator const& other) const noexcept {
    return m_begin > other.m_begin;
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr bool operator<=(inner_iterator const& other) const noexcept {
    return m_begin <= other.m_begin;
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr bool operator>=(inner_iterator const& other) const noexcept {
    return m_begin >= other.m_begin;
  }
};

template <class Iterator, class OuterIndex, class InnerIndex>
class inner_iterator<Iterator, layout::left, OuterIndex, InnerIndex> {
  Iterator m_begin;
  OuterIndex m_outer_size;
 public:
  using value_type = typename Iterator::value_type;
  using size_type = InnerIndex;
  using difference_type = InnerIndex;
  using reference = typename Iterator::reference;
  using pointer = typename Iterator::pointer;
  using iterator_category = std::random_access_iterator_tag;
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr inner_iterator(Iterator const& begin_in,
      OuterIndex const& outer_size_in) noexcept
    :m_begin(begin_in)
    ,m_outer_size(outer_size_in)
  {
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr bool operator==(inner_iterator const& other) const noexcept {
    return m_begin == other.m_begin;
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr bool operator!=(inner_iterator const& other) const noexcept {
    return m_begin != other.m_begin;
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr reference operator*() const noexcept {
    return *m_begin;
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE inner_iterator& operator++() noexcept {
    m_begin += (m_outer_size * difference_type(1));
    return *this;
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE inner_iterator operator++(int) noexcept {
    auto ret = *this;
    m_begin += (m_outer_size * difference_type(1));
    return ret;
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE inner_iterator& operator--() noexcept {
    m_begin -= (m_outer_size * difference_type(1));
    return *this;
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE inner_iterator operator--(int) noexcept {
    auto ret = *this;
    m_begin -= (m_outer_size * difference_type(1));
    return ret;
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE inner_iterator& operator+=(difference_type const n) noexcept {
    m_begin += (m_outer_size * n);
    return *this;
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE inner_iterator& operator-=(difference_type const n) noexcept {
    m_begin -= (m_outer_size * n);
    return *this;
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr inner_iterator operator+(difference_type const n) const noexcept {
    return inner_iterator(m_begin + (m_outer_size * n), m_outer_size);
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr inner_iterator operator-(difference_type const n) const noexcept {
    return inner_iterator(m_begin - (m_outer_size * n), m_outer_size);
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr reference operator[](difference_type const n) const noexcept {
    return *((*this) + n);
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr bool operator<(inner_iterator const& other) const noexcept {
    return m_begin < other.m_begin;
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr bool operator>(inner_iterator const& other) const noexcept {
    return m_begin > other.m_begin;
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr bool operator<=(inner_iterator const& other) const noexcept {
    return m_begin <= other.m_begin;
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr bool operator>=(inner_iterator const& other) const noexcept {
    return m_begin >= other.m_begin;
  }
};

template <class Iterator, class OuterIndex, class InnerIndex>
class outer_iterator<Iterator, layout::right, OuterIndex, InnerIndex> {
  Iterator m_begin;
  InnerIndex m_inner_size;
 public:
  using inner_iterator_type = inner_iterator<Iterator, layout::right, OuterIndex, InnerIndex>;
  using inner_difference_type = typename inner_iterator_type::difference_type;
  using value_type = ::hpc::iterator_range<inner_iterator_type>;
  using difference_type = OuterIndex;
  using reference = value_type;
  using pointer = value_type const*;
  using iterator_category = std::random_access_iterator_tag;
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr outer_iterator(Iterator const& begin_in,
      InnerIndex const& inner_size_in)
    :m_begin(begin_in)
    ,m_inner_size(inner_size_in)
  {
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr bool operator==(outer_iterator const& other) const noexcept {
    return m_begin == other.m_begin;
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr bool operator!=(outer_iterator const& other) const noexcept {
    return m_begin != other.m_begin;
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr reference operator*() const noexcept {
    return reference(
        inner_iterator_type(m_begin),
        inner_iterator_type(m_begin + (difference_type(1) * m_inner_size)));
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE outer_iterator& operator++() noexcept {
    m_begin += inner_difference_type(difference_type(1) * m_inner_size);
    return *this;
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE outer_iterator operator++(int) noexcept {
    auto ret = *this;
    m_begin += inner_difference_type(difference_type(1) * m_inner_size);
    return ret;
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE outer_iterator& operator--() noexcept {
    m_begin -= inner_difference_type(difference_type(1) * m_inner_size);
    return *this;
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE outer_iterator operator--(int) noexcept {
    auto ret = *this;
    m_begin -= inner_difference_type(difference_type(1) * m_inner_size);
    return ret;
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE outer_iterator& operator+=(difference_type const n) noexcept {
    m_begin += (n * m_inner_size);
    return *this;
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE outer_iterator& operator-=(difference_type const n) noexcept {
    m_begin -= (n * m_inner_size);
    return *this;
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr outer_iterator operator+(difference_type const n) const noexcept {
    return outer_iterator(m_begin + inner_difference_type(n * m_inner_size), m_inner_size);
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr outer_iterator operator-(difference_type const n) const noexcept {
    return outer_iterator(m_begin - inner_difference_type(n * m_inner_size), m_inner_size);
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr reference operator[](difference_type const n) const noexcept {
    return *((*this) + n);
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr bool operator<(outer_iterator const& other) const noexcept {
    return m_begin < other.m_begin;
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr bool operator>(outer_iterator const& other) const noexcept {
    return m_begin > other.m_begin;
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr bool operator<=(outer_iterator const& other) const noexcept {
    return m_begin <= other.m_begin;
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr bool operator>=(outer_iterator const& other) const noexcept {
    return m_begin >= other.m_begin;
  }
};

template <class Iterator, class OuterIndex, class InnerIndex>
class outer_iterator<Iterator, layout::left, OuterIndex, InnerIndex> {
  Iterator m_begin;
  OuterIndex m_outer_size;
  InnerIndex m_inner_size;
 public:
  using inner_iterator_type = inner_iterator<Iterator, layout::left, OuterIndex, InnerIndex>;
  using value_type = ::hpc::iterator_range<inner_iterator_type>;
  using difference_type = OuterIndex;
  using reference = value_type;
  using pointer = value_type const*;
  using iterator_category = std::random_access_iterator_tag;
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr explicit outer_iterator(Iterator const& begin_in,
      OuterIndex const& outer_size_in,
      InnerIndex const& inner_size_in)
    :m_begin(begin_in)
    ,m_outer_size(outer_size_in)
    ,m_inner_size(inner_size_in)
  {
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr bool operator==(outer_iterator const& other) const noexcept {
    return m_begin == other.m_begin;
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr bool operator!=(outer_iterator const& other) const noexcept {
    return m_begin != other.m_begin;
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr reference operator*() const noexcept {
    return reference(inner_iterator_type(m_begin, m_outer_size),
        inner_iterator_type(m_begin + m_outer_size * m_inner_size, m_outer_size));
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE outer_iterator& operator++() noexcept {
    m_begin += (OuterIndex(1) * InnerIndex(1));
    return *this;
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE outer_iterator operator++(int) noexcept {
    auto ret = *this;
    m_begin += (OuterIndex(1) * InnerIndex(1));
    return ret;
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE outer_iterator& operator--() noexcept {
    m_begin -= (OuterIndex(1) * InnerIndex(1));
    return *this;
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE outer_iterator operator--(int) noexcept {
    auto ret = *this;
    m_begin -= (OuterIndex(1) * InnerIndex(1));
    return ret;
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE outer_iterator& operator+=(difference_type const n) noexcept {
    m_begin += (n * InnerIndex(1));
    return *this;
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE outer_iterator& operator-=(difference_type const n) noexcept {
    m_begin -= (n * InnerIndex(1));
    return *this;
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr outer_iterator operator+(difference_type const n) const noexcept {
    return outer_iterator(m_begin + (n * InnerIndex(1)), m_outer_size, m_inner_size);
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr outer_iterator operator-(difference_type const n) const noexcept {
    return outer_iterator(m_begin - (n * InnerIndex(1)), m_outer_size, m_inner_size);
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr reference operator[](difference_type const n) const noexcept {
    return *((*this) + n);
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr bool operator<(outer_iterator const& other) const noexcept {
    return m_begin < other.m_begin;
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr bool operator>(outer_iterator const& other) const noexcept {
    return m_begin > other.m_begin;
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr bool operator<=(outer_iterator const& other) const noexcept {
    return m_begin <= other.m_begin;
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr bool operator>=(outer_iterator const& other) const noexcept {
    return m_begin >= other.m_begin;
  }
};

}

template <class Iterator, layout L, class OuterIndex, class InnerIndex>
class range_product;

template <class Iterator, class OuterIndex, class InnerIndex>
class range_product<Iterator, layout::right, OuterIndex, InnerIndex> {
  Iterator m_begin;
  OuterIndex m_outer_size;
  InnerIndex m_inner_size;
 public:
  using iterator = ::hpc::impl::outer_iterator<Iterator, layout::right, OuterIndex, InnerIndex>;
  using const_iterator = iterator;
  using value_type = typename iterator::value_type;
  using size_type = OuterIndex;
  using difference_type = OuterIndex;
  using reference = typename iterator::reference;
  using const_reference = typename iterator::reference;
  using pointer = typename iterator::pointer;
  using const_pointer = typename iterator::pointer;
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr range_product(
      Iterator const& begin_in,
      OuterIndex outer_size_in,
      InnerIndex inner_size_in)
    :m_begin(begin_in)
    ,m_outer_size(outer_size_in)
    ,m_inner_size(inner_size_in)
  {
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr const_iterator begin() const noexcept {
    return iterator(m_begin, m_inner_size);
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr const_iterator cbegin() const noexcept {
    return iterator(m_begin, m_inner_size);
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr const_iterator end() const noexcept {
    return iterator(m_begin + m_outer_size * m_inner_size, m_inner_size);
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr const_iterator cend() const noexcept {
    return iterator(m_begin + m_outer_size * m_inner_size, m_inner_size);
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr bool empty() const noexcept { return size() == 0; }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr size_type size() const noexcept { return m_outer_size; }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr value_type operator[](difference_type const n) const noexcept {
    return begin()[n];
  }
};

template <class Iterator, class OuterIndex, class InnerIndex>
class range_product<Iterator, layout::left, OuterIndex, InnerIndex> {
  Iterator m_begin;
  OuterIndex m_outer_size;
  InnerIndex m_inner_size;
 public:
  using iterator = ::hpc::impl::outer_iterator<Iterator, layout::left, OuterIndex, InnerIndex>;
  using const_iterator = iterator;
  using value_type = typename iterator::value_type;
  using size_type = OuterIndex;
  using difference_type = OuterIndex;
  using reference = typename iterator::reference;
  using const_reference = typename iterator::reference;
  using pointer = typename iterator::pointer;
  using const_pointer = typename iterator::pointer;
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr range_product(
      Iterator const& begin_in,
      OuterIndex outer_size_in,
      InnerIndex inner_size_in)
    noexcept
    :m_begin(begin_in)
    ,m_outer_size(outer_size_in)
    ,m_inner_size(inner_size_in)
  {
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr const_iterator begin() const noexcept {
    return iterator(m_begin, m_outer_size, m_inner_size);
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr const_iterator cbegin() const noexcept {
    return iterator(m_begin, m_outer_size, m_inner_size);
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr const_iterator end() const noexcept {
    return iterator(m_begin + m_outer_size * InnerIndex(1), m_outer_size, m_inner_size);
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr const_iterator cend() const noexcept {
    return iterator(m_begin + m_outer_size * InnerIndex(1), m_outer_size, m_inner_size);
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr bool empty() const noexcept { return size() == 0; }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr size_type size() const noexcept { return m_outer_size; }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr value_type operator[](difference_type const n) const noexcept {
    return begin()[n];
  }
};

static constexpr layout host_layout = layout::right;
static constexpr layout device_layout = layout::right;

template <layout L = ::hpc::device_layout, class O = std::ptrdiff_t, class I = std::ptrdiff_t>
using counting_product = ::hpc::range_product<::hpc::counting_iterator<decltype(O() * I())>, L, O, I>;

template <layout L = ::hpc::device_layout, class O = std::ptrdiff_t, class I = std::ptrdiff_t>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr
auto
make_counting_product(O outer_count, I inner_count) noexcept {
  using product_index = decltype(O() * I());
  using product_iterator = counting_iterator<product_index>;
  return counting_product<L, O, I>(
      product_iterator(product_index(0)),
      outer_count, inner_count);
}

template <class O, class I>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr
auto
operator*(counting_range<O> const& a, counting_range<I> const& b) noexcept {
  return make_counting_product(a.size(), b.size());
}

}
