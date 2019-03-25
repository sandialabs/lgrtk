#pragma once

#include <memory>
#include <type_traits>

namespace lgr {

template <class T, class Index = int>
class vector_iterator {
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
    m_ptr += n;
    return *this;
  }
  inline vector_iterator& operator-=(difference_type const n) noexcept {
    m_ptr -= n;
    return *this;
  }
  inline vector_iterator operator+(difference_type const n) const noexcept {
    return vector_iterator(m_ptr + n);
  }
  inline vector_iterator operator-(difference_type const n) const noexcept {
    return vector_iterator(m_ptr - n);
  }
  inline difference_type operator-(vector_iterator const& other) const noexcept {
    return difference_type(m_ptr - other.m_ptr);
  }
  inline reference operator[](Index const i) const noexcept {
    return *(m_ptr + (i - Index(0)));
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

template <class T, class Allocator>
class vector {
protected:
  using allocator_traits_type = std::allocator_traits<Allocator>;
  Allocator m_allocator;
  T* m_data;
  std::size_t m_size;
public:
  using value_type = T;
  using allocator_type = Allocator;
  using size_type = int;
  using difference_type = int;
  using reference = value_type&;
  using const_reference = value_type const&;
  using pointer = typename allocator_traits_type::pointer;
  using const_pointer = typename allocator_traits_type::const_pointer;
  using iterator = vector_iterator<T, size_type>;
  using const_iterator = vector_iterator<T const, size_type>;
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
  iterator end() noexcept { return iterator(m_data + m_size); }
  const_iterator end() const noexcept { return const_iterator(m_data + m_size); }
  const_iterator cend() const noexcept { return const_iterator(m_data + m_size); }
  bool empty() const noexcept { return m_size == 0; }
  size_type size() const noexcept { return size_type(m_size); }
  void clear() {
    if (m_data) allocator_traits_type::deallocate(m_allocator, m_data, m_size);
    m_data = nullptr;
    m_size = 0;
  }
  void resize(size_type count) {
    if (m_size == std::size_t(count)) return;
    clear();
    m_data = allocator_traits_type::allocate(m_allocator, std::size_t(count));
    m_size = std::size_t(count);
  }
  void swap(vector& other) {
    using std::swap;
    swap(m_allocator, other.m_allocator);
    swap(m_data, other.m_data);
    swap(m_size, other.m_size);
  }
  allocator_type get_allocator() const { return m_allocator; }
};

template <class T, class Allocator>
void swap(vector<T, Allocator>&, vector<T, Allocator>&);

}
