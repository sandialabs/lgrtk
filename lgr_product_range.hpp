#pragma once

#include <lgr_iterator_range.hpp>
#include <lgr_layout.hpp>

namespace lgr {

template <class Iterator, layout L, class OuterIndex, class InnerIndex>
class inner_iterator; 

template <class Iterator, layout L, class OuterIndex, class InnerIndex>
class outer_iterator; 

template <class Iterator, layout L, class OuterIndex, class InnerIndex>
class product_range; 

template <class Iterator, class OuterIndex, class InnerIndex>
class inner_iterator<Iterator, AOS, OuterIndex, InnerIndex> {
  Iterator m_begin;
  public:
  using value_type = typename Iterator::value_type;
  using size_type = InnerIndex;
  using difference_type = InnerIndex;
  using reference = typename Iterator::reference;
  using pointer = typename Iterator::pointer;
  using iterator_category = std::random_access_iterator_tag;
  explicit inline inner_iterator(Iterator const& impl_in)
    :m_begin(impl_in)
  {
  }
  inline bool operator==(inner_iterator const& other) const noexcept {
    return m_begin == other.m_begin;
  }
  inline bool operator!=(inner_iterator const& other) const noexcept {
    return m_begin != other.m_begin;
  }
  inline reference operator*() const noexcept {
    return *m_begin;
  }
  inline inner_iterator& operator++() noexcept {
    ++m_begin;
    return *this;
  }
  inline inner_iterator operator++(int) noexcept {
    auto ret = *this;
    ++m_begin;
    return ret;
  }
  inline inner_iterator& operator--() noexcept {
    --m_begin;
    return *this;
  }
  inline inner_iterator operator--(int) noexcept {
    auto ret = *this;
    --m_begin;
    return ret;
  }
  inline inner_iterator& operator+=(difference_type const n) noexcept {
    m_begin += OuterIndex(1) * n;
    return *this;
  }
  inline inner_iterator& operator-=(difference_type const n) noexcept {
    m_begin -= OuterIndex(1) * n;
    return *this;
  }
  inline inner_iterator operator+(difference_type const n) const noexcept {
    return inner_iterator(m_begin + (OuterIndex(1) * n));
  }
  inline inner_iterator operator-(difference_type const n) const noexcept {
    return inner_iterator(m_begin - (OuterIndex(1) * n));
  }
  inline difference_type operator-(inner_iterator const& other) const noexcept {
    return difference_type(m_begin - other.m_begin);
  }
  inline reference operator[](difference_type const n) const noexcept {
    return m_begin[OuterIndex(1) * n];
  }
  inline bool operator<(inner_iterator const& other) const noexcept {
    return m_begin < other.m_begin;
  }
  inline bool operator>(inner_iterator const& other) const noexcept {
    return m_begin > other.m_begin;
  }
  inline bool operator<=(inner_iterator const& other) const noexcept {
    return m_begin <= other.m_begin;
  }
  inline bool operator>=(inner_iterator const& other) const noexcept {
    return m_begin >= other.m_begin;
  }
};

template <class Iterator, class OuterIndex, class InnerIndex>
class inner_iterator<Iterator, SOA, OuterIndex, InnerIndex> {
  Iterator m_begin;
  OuterIndex m_outer_size;
  public:
  using value_type = typename Iterator::value_type;
  using size_type = InnerIndex;
  using difference_type = InnerIndex;
  using reference = typename Iterator::reference;
  using pointer = typename Iterator::pointer;
  using iterator_category = std::random_access_iterator_tag;
  explicit inline inner_iterator(Iterator const& begin_in,
      OuterIndex const& outer_size_in)
    :m_begin(begin_in)
    ,m_outer_size(outer_size_in)
  {
  }
  inline bool operator==(inner_iterator const& other) const noexcept {
    return m_begin == other.m_begin;
  }
  inline bool operator!=(inner_iterator const& other) const noexcept {
    return m_begin != other.m_begin;
  }
  inline reference operator*() const noexcept {
    return *m_begin;
  }
  inline inner_iterator& operator++() noexcept {
    m_begin += (m_outer_size * difference_type(1));
    return *this;
  }
  inline inner_iterator operator++(int) noexcept {
    auto ret = *this;
    m_begin += (m_outer_size * difference_type(1));
    return ret;
  }
  inline inner_iterator& operator--() noexcept {
    m_begin -= (m_outer_size * difference_type(1));
    return *this;
  }
  inline inner_iterator operator--(int) noexcept {
    auto ret = *this;
    m_begin -= (m_outer_size * difference_type(1));
    return ret;
  }
  inline inner_iterator& operator+=(difference_type const n) noexcept {
    m_begin += (m_outer_size * n);
    return *this;
  }
  inline inner_iterator& operator-=(difference_type const n) noexcept {
    m_begin -= (m_outer_size * n);
    return *this;
  }
  inline inner_iterator operator+(difference_type const n) const noexcept {
    return inner_iterator(m_begin + (m_outer_size * n), m_outer_size);
  }
  inline inner_iterator operator-(difference_type const n) const noexcept {
    return inner_iterator(m_begin - (m_outer_size * n), m_outer_size);
  }
  inline reference operator[](difference_type const n) const noexcept {
    return m_begin[m_outer_size * n];
  }
  inline bool operator<(inner_iterator const& other) const noexcept {
    return m_begin < other.m_begin;
  }
  inline bool operator>(inner_iterator const& other) const noexcept {
    return m_begin > other.m_begin;
  }
  inline bool operator<=(inner_iterator const& other) const noexcept {
    return m_begin <= other.m_begin;
  }
  inline bool operator>=(inner_iterator const& other) const noexcept {
    return m_begin >= other.m_begin;
  }
};

template <class Iterator, class OuterIndex, class InnerIndex>
class outer_iterator<Iterator, AOS, OuterIndex, InnerIndex> {
  Iterator m_begin;
  InnerIndex m_inner_size;
  public:
  using inner_iterator_type = inner_iterator<Iterator, AOS, OuterIndex, InnerIndex>;
  using value_type = iterator_range<inner_iterator_type>;
  using difference_type = OuterIndex;
  using reference = value_type;
  using pointer = value_type const*;
  using iterator_category = std::random_access_iterator_tag;
  explicit inline outer_iterator(Iterator const& begin_in,
      InnerIndex const& inner_size_in)
    :m_begin(begin_in)
    ,m_inner_size(inner_size_in)
  {
  }
  inline bool operator==(outer_iterator const& other) const noexcept {
    return m_begin == other.m_begin;
  }
  inline bool operator!=(outer_iterator const& other) const noexcept {
    return m_begin != other.m_begin;
  }
  inline reference operator*() const noexcept {
    return reference(
        inner_iterator_type(m_begin),
        inner_iterator_type(m_begin + (difference_type(1) * m_inner_size)));
  }
  inline outer_iterator& operator++() noexcept {
    m_begin += (difference_type(1) * m_inner_size);
    return *this;
  }
  inline outer_iterator operator++(int) noexcept {
    auto ret = *this;
    m_begin += (difference_type(1) * m_inner_size);
    return ret;
  }
  inline outer_iterator& operator--() noexcept {
    m_begin -= (difference_type(1) * m_inner_size);
    return *this;
  }
  inline outer_iterator operator--(int) noexcept {
    auto ret = *this;
    m_begin -= (difference_type(1) * m_inner_size);
    return ret;
  }
  inline outer_iterator& operator+=(difference_type const n) noexcept {
    m_begin += (n * m_inner_size);
    return *this;
  }
  inline outer_iterator& operator-=(difference_type const n) noexcept {
    m_begin -= (n * m_inner_size);
    return *this;
  }
  inline outer_iterator operator+(difference_type const n) const noexcept {
    return outer_iterator(m_begin + (n * m_inner_size), m_inner_size);
  }
  inline outer_iterator operator-(difference_type const n) const noexcept {
    return outer_iterator(m_begin - (n * m_inner_size), m_inner_size);
  }
  inline reference operator[](difference_type const n) const noexcept {
    return *(operator+(n));
  }
  inline bool operator<(outer_iterator const& other) const noexcept {
    return m_begin < other.m_begin;
  }
  inline bool operator>(outer_iterator const& other) const noexcept {
    return m_begin > other.m_begin;
  }
  inline bool operator<=(outer_iterator const& other) const noexcept {
    return m_begin <= other.m_begin;
  }
  inline bool operator>=(outer_iterator const& other) const noexcept {
    return m_begin >= other.m_begin;
  }
};

template <class Iterator, class OuterIndex, class InnerIndex>
class outer_iterator<Iterator, SOA, OuterIndex, InnerIndex> {
  Iterator m_begin;
  OuterIndex m_outer_size;
  InnerIndex m_inner_size;
  public:
  using inner_iterator_type = inner_iterator<Iterator, SOA, OuterIndex, InnerIndex>;
  using value_type = iterator_range<inner_iterator_type>;
  using difference_type = OuterIndex;
  using reference = value_type;
  using pointer = value_type const*;
  using iterator_category = std::random_access_iterator_tag;
  explicit inline outer_iterator(Iterator const& begin_in,
      OuterIndex const& outer_size_in,
      InnerIndex const& inner_size_in)
    :m_begin(begin_in)
    ,m_outer_size(outer_size_in)
    ,m_inner_size(inner_size_in)
  {
  }
  inline bool operator==(outer_iterator const& other) const noexcept {
    return m_begin == other.m_begin;
  }
  inline bool operator!=(outer_iterator const& other) const noexcept {
    return m_begin != other.m_begin;
  }
  inline reference operator*() const noexcept {
    return reference(inner_iterator_type(m_begin, m_outer_size),
        inner_iterator_type(m_begin + m_outer_size * m_inner_size, m_outer_size));
  }
  inline outer_iterator& operator++() noexcept {
    m_begin += (OuterIndex(1) * InnerIndex(1));
    return *this;
  }
  inline outer_iterator operator++(int) noexcept {
    auto ret = *this;
    m_begin += (OuterIndex(1) * InnerIndex(1));
    return ret;
  }
  inline outer_iterator& operator--() noexcept {
    m_begin -= (OuterIndex(1) * InnerIndex(1));
    return *this;
  }
  inline outer_iterator operator--(int) noexcept {
    auto ret = *this;
    m_begin -= (OuterIndex(1) * InnerIndex(1));
    return ret;
  }
  inline outer_iterator& operator+=(difference_type const n) noexcept {
    m_begin += (n * InnerIndex(1));
    return *this;
  }
  inline outer_iterator& operator-=(difference_type const n) noexcept {
    m_begin -= (n * InnerIndex(1));
    return *this;
  }
  inline outer_iterator operator+(difference_type const n) const noexcept {
    return outer_iterator(m_begin + (n * InnerIndex(1)), m_outer_size);
  }
  inline outer_iterator operator-(difference_type const n) const noexcept {
    return outer_iterator(m_begin - (n * InnerIndex(1)), m_outer_size);
  }
  inline reference operator[](difference_type const n) const noexcept {
    return *(operator+(n));
  }
  inline bool operator<(outer_iterator const& other) const noexcept {
    return m_begin < other.m_begin;
  }
  inline bool operator>(outer_iterator const& other) const noexcept {
    return m_begin > other.m_begin;
  }
  inline bool operator<=(outer_iterator const& other) const noexcept {
    return m_begin <= other.m_begin;
  }
  inline bool operator>=(outer_iterator const& other) const noexcept {
    return m_begin >= other.m_begin;
  }
};

template <class Iterator, class OuterIndex, class InnerIndex>
class product_range<Iterator, AOS, OuterIndex, InnerIndex> {
  Iterator m_begin;
  OuterIndex m_outer_size;
  InnerIndex m_inner_size;
  public:
  using iterator = outer_iterator<Iterator, AOS, OuterIndex, InnerIndex>;
  using const_iterator = iterator;
  using value_type = typename iterator::value_type;
  using size_type = OuterIndex;
  using difference_type = OuterIndex;
  using reference = typename iterator::reference;
  using const_reference = typename iterator::reference;
  using pointer = typename iterator::pointer;
  using const_pointer = typename iterator::pointer;
  explicit inline product_range(
      Iterator const& begin_in,
      OuterIndex outer_size_in,
      InnerIndex inner_size_in)
    :m_begin(begin_in)
    ,m_outer_size(outer_size_in)
    ,m_inner_size(inner_size_in)
  {
  }
  inline const_iterator begin() const noexcept {
    return iterator(m_begin, m_inner_size);
  }
  inline const_iterator cbegin() const noexcept {
    return iterator(m_begin, m_inner_size);
  }
  inline const_iterator end() const noexcept {
    return iterator(m_begin + m_outer_size * m_inner_size, m_inner_size);
  }
  inline const_iterator cend() const noexcept {
    return iterator(m_begin + m_outer_size * m_inner_size, m_inner_size);
  }
  inline bool empty() { return size() == 0; }
  inline size_type size() { return m_outer_size; }
  inline value_type operator[](difference_type const n) const noexcept {
    return begin()[n];
  }
};

template <class Iterator, class OuterIndex, class InnerIndex>
class product_range<Iterator, SOA, OuterIndex, InnerIndex> {
  Iterator m_begin;
  OuterIndex m_outer_size;
  InnerIndex m_inner_size;
  public:
  using iterator = outer_iterator<Iterator, SOA, OuterIndex, InnerIndex>;
  using const_iterator = iterator;
  using value_type = typename iterator::value_type;
  using size_type = OuterIndex;
  using difference_type = OuterIndex;
  using reference = typename iterator::reference;
  using const_reference = typename iterator::reference;
  using pointer = typename iterator::pointer;
  using const_pointer = typename iterator::pointer;
  explicit inline product_range(
      Iterator const& begin_in,
      OuterIndex outer_size_in,
      InnerIndex inner_size_in)
    :m_begin(begin_in)
    ,m_outer_size(outer_size_in)
    ,m_inner_size(inner_size_in)
  {
  }
  inline const_iterator begin() const noexcept {
    return iterator(m_begin, m_outer_size, m_inner_size);
  }
  inline const_iterator cbegin() const noexcept {
    return iterator(m_begin, m_outer_size, m_inner_size);
  }
  inline const_iterator end() const noexcept {
    return iterator(m_begin + m_outer_size * InnerIndex(1), m_outer_size, m_inner_size);
  }
  inline const_iterator cend() const noexcept {
    return iterator(m_begin + m_outer_size * InnerIndex(1), m_outer_size, m_inner_size);
  }
  inline bool empty() { return size() == 0; }
  inline size_type size() { return m_outer_size; }
  inline value_type operator[](difference_type const n) const noexcept {
    return begin()[n];
  }
};

}
