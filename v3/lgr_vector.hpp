#pragma once

#include <memory>
#include <type_traits>
#include <lgr_pointer_iterator.hpp>

namespace lgr {

template <class T, class Allocator, class Index = int>
class vector {
protected:
  using allocator_traits_type = std::allocator_traits<Allocator>;
  Allocator m_allocator;
  T* m_data;
  Index m_size;
public:
  using value_type = T;
  using allocator_type = Allocator;
  using size_type = Index;
  using difference_type = decltype(m_size - m_size);
  using reference = value_type&;
  using const_reference = value_type const&;
  using pointer = typename allocator_traits_type::pointer;
  using const_pointer = typename allocator_traits_type::const_pointer;
  using iterator = pointer_iterator<T, size_type>;
  using const_iterator = pointer_iterator<T const, size_type>;
  explicit vector(Allocator const& allocator_in) noexcept
    :m_allocator(allocator_in)
    ,m_data(nullptr)
    ,m_size(0)
  {}
  explicit vector(size_type count, Allocator const& allocator_in)
    :m_allocator(allocator_in)
    ,m_data(nullptr)
    ,m_size(0)
  {
    resize(count);
  }
  vector(vector&&) noexcept = default;
  vector(vector const&) = delete;
  vector& operator=(vector const&) = delete;
  vector& operator=(vector&&) = delete;
  ~vector() { clear(); }
  T* data() noexcept { return m_data; }
  T const* data() const noexcept { return m_data; }
  iterator begin() noexcept { return iterator(m_data); }
  const_iterator begin() const noexcept { return const_iterator(m_data); }
  const_iterator cbegin() const noexcept { return const_iterator(m_data); }
  iterator end() noexcept { return iterator(m_data + int(m_size)); }
  const_iterator end() const noexcept { return const_iterator(m_data + int(m_size)); }
  const_iterator cend() const noexcept { return const_iterator(m_data + int(m_size)); }
  bool empty() const noexcept { return m_size == 0; }
  size_type size() const noexcept { return m_size; }
  void clear() {
    if (m_data) allocator_traits_type::deallocate(m_allocator, m_data, std::size_t(m_size));
    m_data = nullptr;
    m_size = size_type(0);
  }
  void resize(size_type count) {
    if (m_size == count) return;
    clear();
    m_data = allocator_traits_type::allocate(m_allocator, std::size_t(count));
    m_size = count;
  }
  void swap(vector& other) {
    using std::swap;
    swap(m_allocator, other.m_allocator);
    swap(m_data, other.m_data);
    swap(m_size, other.m_size);
  }
  allocator_type get_allocator() const { return m_allocator; }
};

}
