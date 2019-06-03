#pragma once

#include <cassert>
#include <iterator>

#include <hpc_range.hpp>
#include <hpc_vector.hpp>
#include <hpc_numeric.hpp>

namespace hpc {

template <class TargetIndex, class SourceIndex>
class range_sum_iterator {
  TargetIndex const* m_ptr;

 public:
  using value_type = counting_range<TargetIndex>;
  using difference_type = SourceIndex;
  using reference = value_type;
  using pointer = value_type const*;
  using iterator_category = std::random_access_iterator_tag;
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE explicit constexpr range_sum_iterator(TargetIndex const* ptr_in) noexcept : m_ptr(ptr_in) {}
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr bool operator==(range_sum_iterator const& other) const noexcept {
    return m_ptr == other.m_ptr;
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr bool operator!=(range_sum_iterator const& other) const noexcept {
    return m_ptr != other.m_ptr;
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr reference operator*() const noexcept {
    return value_type(m_ptr[0], m_ptr[1]);
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE range_sum_iterator& operator++() noexcept {
    ++m_ptr;
    return *this;
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE range_sum_iterator operator++(int) noexcept {
    auto const ret = *this;
    ++m_ptr;
    return ret;
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE range_sum_iterator& operator--() noexcept {
    --m_ptr;
    return *this;
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE range_sum_iterator operator--(int) noexcept {
    auto ret = *this;
    --m_ptr;
    return ret;
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE range_sum_iterator& operator+=(difference_type const n) noexcept {
    m_ptr += std::ptrdiff_t(n);
    return *this;
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE range_sum_iterator& operator-=(difference_type const n) noexcept {
    m_ptr -= std::ptrdiff_t(n);
    return *this;
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr range_sum_iterator operator+(difference_type const n) const noexcept {
    return range_sum_iterator(m_ptr + std::ptrdiff_t(n));
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr range_sum_iterator operator-(difference_type const n) const noexcept {
    return range_sum_iterator(m_ptr - std::ptrdiff_t(n));
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr difference_type operator-(range_sum_iterator const& other) const noexcept {
    return difference_type(m_ptr - other.m_ptr);
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr reference operator[](difference_type const n) const noexcept {
    return *((*this) + n);
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr bool operator<(range_sum_iterator const& other) const noexcept {
    return m_ptr < other.m_ptr;
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr bool operator>(range_sum_iterator const& other) const noexcept {
    return m_ptr > other.m_ptr;
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr bool operator<=(range_sum_iterator const& other) const noexcept {
    return m_ptr <= other.m_ptr;
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr bool operator>=(range_sum_iterator const& other) const noexcept {
    return m_ptr >= other.m_ptr;
  }
};

template <class TargetIndex, class Allocator, class ExecutionPolicy, class SourceIndex>
class range_sum {
  using vector_type = ::hpc::vector<TargetIndex, Allocator, ExecutionPolicy, SourceIndex>;
  vector_type m_vector;
public:
  using value_type = ::hpc::counting_range<TargetIndex>;
  using allocator_type = typename vector_type::allocator_type;
  using execution_policy = typename vector_type::execution_policy;
  using size_type = typename vector_type::size_type;
  using difference_type = typename vector_type::difference_type;
  using reference = value_type;
  using const_reference = value_type;
  using pointer = TargetIndex*;
  using const_pointer = TargetIndex const*;
  using iterator = ::hpc::range_sum_iterator<TargetIndex, SourceIndex>;
  using const_iterator = iterator;
  constexpr range_sum() noexcept : m_vector() {}
  template <class RangeSizes>
  explicit range_sum(RangeSizes const& range_sizes)
  {
    assign_sizes(range_sizes);
  }
  constexpr range_sum(allocator_type const& allocator_in, execution_policy const& policy_in) noexcept
    :m_vector(allocator_in, policy_in)
  {}
  template <class RangeSizes>
  range_sum(RangeSizes const& range_sizes, allocator_type const& allocator_in, execution_policy const& policy_in)
    :m_vector(allocator_in, policy_in)
  {
    assign_sizes(range_sizes);
  }
  range_sum(range_sum&&) noexcept = default;
  range_sum& operator=(range_sum&&) = default;
  range_sum(range_sum const&) = delete;
  range_sum& operator=(range_sum const&) = delete;
  const_iterator begin() const noexcept {
    return iterator(m_vector.data());
  }
  const_iterator cbegin() const noexcept {
    return iterator(m_vector.data());
  }
  const_iterator end() const noexcept {
    return iterator(m_vector.data() + size());
  }
  const_iterator cend() const noexcept {
    return iterator(m_vector.data() + size());
  }
  const_reference operator[](size_type i) const noexcept {
    return begin()[i];
  }
  bool empty() const noexcept {
    return m_vector.size() <= 1;
  }
  size_type size() const noexcept {
    return ::hpc::max(size_type(1), m_vector.size()) - size_type(1);
  }
  template <class RangeSizes>
  void assign_sizes(RangeSizes const& sizes) {
    m_vector.resize(sizes.size() + size_type(1));
    auto it = m_vector.begin();
    auto const first = it;
    ++it;
    auto const second = it;
    auto const first_range = ::hpc::iterator_range<decltype(it)>(first, second);
    using subrange_size_type = typename RangeSizes::value_type;
    ::hpc::fill(get_execution_policy(), first_range, TargetIndex(0));
    auto const end = m_vector.end();
    auto const rest = iterator_range<decltype(it)>(second, end);
    auto const unop = [] HPC_HOST_DEVICE (subrange_size_type const i) { return TargetIndex(std::ptrdiff_t(i)); };
    ::hpc::transform_inclusive_scan(get_execution_policy(), sizes, rest, ::hpc::plus<TargetIndex>(), unop);
  }
  allocator_type get_allocator() const noexcept { return m_vector.get_allocator(); }
  execution_policy get_execution_policy() const noexcept { return m_vector.get_execution_policy(); }
};

template <class T = std::ptrdiff_t, class S = std::ptrdiff_t>
using host_range_sum = range_sum<T, ::hpc::host_allocator<T>, ::hpc::host_policy, S>;
template <class T = std::ptrdiff_t, class S = std::ptrdiff_t>
using device_range_sum = range_sum<T, ::hpc::device_allocator<T>, ::hpc::device_policy, S>;
template <class T = std::ptrdiff_t, class S = std::ptrdiff_t>
using pinned_range_sum = range_sum<T, ::hpc::pinned_allocator<T>, ::hpc::host_policy, S>;

}
