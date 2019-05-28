#pragma once

#include <hpc_range.hpp>

namespace hpc {

template <
  class T,
  layout L = layout::right,
  class Allocator = std::allocator<T>,
  class ExecutionPolicy = ::hpc::serial_policy,
  class RowIndex = std::ptrdiff_t,
  class ColumnIndex = std::ptrdiff_t>
class matrix {
public:
  using row_type = RowIndex;
  using column_type = ColumnIndex;
  using nonzero_type = decltype(std::declval<row_type>() * std::declval<column_type>());
  using value_type = T;
private:
  using data_type = vector<value_type, Allocator, ExecutionPolicy, nonzero_type>;
  using data_iterator = typename data_type::iterator;
  using const_data_iterator = typename data_type::const_iterator;
  using range_type = range_product<data_iterator, L, row_type, column_type>;
  using const_range_type = range_product<const_data_iterator, L, row_type, column_type>;
  vector<value_type, Allocator, ExecutionPolicy, nonzero_type> m_data;
  row_type m_rows;
  constexpr range_type get_range() noexcept {
    return range_type(m_data.begin(), rows(), columns());
  }
  constexpr const_range_type get_range() const noexcept {
    return const_range_type(m_data.begin(), rows(), columns());
  }
public:
  using allocator_type = Allocator;
  using execution_policy = ExecutionPolicy;
  using size_type = row_type;
  using difference_type = decltype(m_rows - m_rows);
  using reference = value_type&;
  using const_reference = value_type const&;
  using pointer = T*;
  using const_pointer = T const*;
  using iterator = typename range_type::iterator;
  using const_iterator = typename range_type::const_iterator;
  constexpr matrix() noexcept = default;
  matrix(row_type row_count, column_type column_count)
    :m_data(row_count * column_count)
    ,m_rows(row_count)
  {
  }
  constexpr matrix(allocator_type const& allocator_in, execution_policy const& exec_in) noexcept
    :m_data(allocator_in, exec_in)
    ,m_rows(0)
  {}
  matrix(row_type row_count, column_type column_count, allocator_type const& allocator_in, execution_policy const& exec_in)
    :m_data(row_count * column_count, allocator_in, exec_in)
    ,m_rows(row_count)
  {
  }
  matrix(matrix&& other) noexcept = default;
  matrix(matrix const&) = delete;
  matrix& operator=(matrix const&) = delete;
  matrix& operator=(matrix&& other) = default;
  ~matrix() = default;
  T* data() noexcept { return m_data.data(); }
  T const* data() const noexcept { return m_data.data(); }
  iterator begin() noexcept {
    return get_range().begin();
  }
  const_iterator begin() const noexcept {
    return get_range().begin();
  }
  const_iterator cbegin() const noexcept {
    return get_range().begin();
  }
  iterator end() noexcept {
    return get_range().end();
  }
  const_iterator end() const noexcept {
    return get_range().end();
  }
  const_iterator cend() const noexcept {
    return get_range().end();
  }
  bool empty() const noexcept { return m_data.empty(); }
  size_type size() const noexcept { return m_rows; }
  row_type rows() const noexcept { return m_rows; }
  column_type columns() const noexcept { return m_data.size() / m_rows; }
  void clear() { m_data.clear(); }
  void resize(row_type row_count, column_type column_count) {
    if (row_count == rows() && column_count == columns()) return;
    m_data.clear();
    m_data.resize(row_count * column_count);
    m_rows = row_count;
  }
  constexpr allocator_type get_allocator() const noexcept { return m_data.get_allocator(); }
  constexpr execution_policy get_execution_policy() const noexcept { return m_data.get_execution_policy(); }
  constexpr typename range_type::reference operator[](size_type i) noexcept { return begin()[i]; }
  constexpr typename range_type::const_reference operator[](size_type i) const noexcept { return begin()[i]; }
  constexpr reference operator()(row_type i, column_type j) noexcept { return begin()[i][j]; }
  constexpr const_reference operator()(row_type i, column_type j) const noexcept { return begin()[i][j]; }
};

}
