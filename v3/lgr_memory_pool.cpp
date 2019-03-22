#include <lgr_memory_pool.hpp>
#include <lgr_fail.hpp>

namespace lgr {

memory_pool::memory_pool(
    decltype(m_malloc) const& malloc_in,
    decltype(m_free) const& free_in)
  :m_malloc(malloc_in)
  ,m_free(free_in)
{
}

memory_pool::~memory_pool() {
  for (int exponent = 0; exponent < 64; ++exponent) {
    for (auto ptr : m_free_list[exponent]) {
      m_free(ptr, std::size_t(1) << exponent);
    }
  }
}

void* memory_pool::allocate(std::size_t size) {
  int exponent = 0;
  while ((std::size_t(1) << exponent) < size) ++exponent;
  auto& freed = m_free_list[exponent];
  if (!freed.empty()) {
    auto it = freed.end();
    --it;
    void* const data = *it;
    freed.erase(it);
    return data;
  }
  std::size_t const actual_size = (std::size_t(1) << exponent);
  return m_malloc(actual_size);
}

void memory_pool::deallocate(void* ptr, std::size_t size) {
  int exponent = 0;
  while ((std::size_t(1) << exponent) < size) ++exponent;
  m_free_list[exponent].push_back(ptr);
}

concurrent_memory_pool::concurrent_memory_pool(
      memory_pool::malloc_type const& malloc_in,
      memory_pool::free_type const& free_in)
  :m_pool(malloc_in, free_in)
  ,m_mutex()
{
}

void* concurrent_memory_pool::allocate(std::size_t size) {
  std::lock_guard<std::mutex> lock(m_mutex);
  return m_pool.allocate(size);
}

void concurrent_memory_pool::deallocate(void* ptr, std::size_t size) {
  std::lock_guard<std::mutex> lock(m_mutex);
  m_pool.deallocate(ptr, size);
}

void* device_malloc(std::size_t size) {
  void* const data = std::malloc(size);
  if (!data) {
    fail("failed to allocate ", size, " bytes in device memory\n");
  }
  return data;
}

void device_free(void* ptr, std::size_t) {
  std::free(ptr);
}

device_memory_pool::device_memory_pool()
  :concurrent_memory_pool(
      device_malloc, device_free)
{
}

void* pinned_malloc(std::size_t size) {
  void* const data = std::malloc(size);
  if (!data) {
    fail("failed to allocate ", size, " bytes in pinned memory\n");
  }
  return data;
}

void pinned_free(void* ptr, std::size_t) {
  std::free(ptr);
}

pinned_memory_pool::pinned_memory_pool()
  :concurrent_memory_pool(
      pinned_malloc, pinned_free)
{
}

}
