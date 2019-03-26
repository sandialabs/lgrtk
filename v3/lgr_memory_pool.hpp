#pragma once

#include <vector>
#include <mutex>
#include <functional>

namespace lgr {

using void_pointer_type = void*;
using malloc_type = std::function<void_pointer_type(std::size_t)>;
using free_type = std::function<void(void_pointer_type, std::size_t)>;

class memory_pool {
  std::vector<void*> m_free_list[64];
private:
  malloc_type m_malloc;
  free_type m_free;
public:
  memory_pool() = delete;
  explicit memory_pool(
      decltype(m_malloc) const& malloc_in,
      decltype(m_free) const& free_in);
  ~memory_pool();
  memory_pool(memory_pool&&) = delete;
  memory_pool(memory_pool const&) = delete;
  memory_pool& operator=(memory_pool&&) = delete;
  memory_pool& operator=(memory_pool const&) = delete;
  void* allocate(std::size_t size);
  void deallocate(void* ptr, std::size_t size);
};

class concurrent_memory_pool {
  memory_pool m_pool;
  std::mutex m_mutex;

public:
  concurrent_memory_pool() = delete;
  explicit concurrent_memory_pool(
      malloc_type const& malloc_in,
      free_type const& free_in);
  void* allocate(std::size_t size);
  void deallocate(void* ptr, std::size_t size);
};

class memory_pool_facade {
private:
  malloc_type m_malloc;
  free_type m_free;
public:
  memory_pool_facade() = delete;
  explicit memory_pool_facade(
      decltype(m_malloc) const& malloc_in,
      decltype(m_free) const& free_in);
  ~memory_pool_facade() = default;
  memory_pool_facade(memory_pool_facade&&) = delete;
  memory_pool_facade(memory_pool_facade const&) = delete;
  memory_pool_facade& operator=(memory_pool_facade&&) = delete;
  memory_pool_facade& operator=(memory_pool_facade const&) = delete;
  void* allocate(std::size_t size);
  void deallocate(void* ptr, std::size_t size);
};

void* device_malloc(std::size_t size);
void device_free(void* ptr, std::size_t size);

using memory_pool_base = memory_pool_facade;

class device_memory_pool : public memory_pool_base {
 public:
  device_memory_pool();
};

void* pinned_malloc(std::size_t size);
void pinned_free(void* ptr, std::size_t size);

class pinned_memory_pool : public memory_pool_base {
public:
  pinned_memory_pool();
};

}
