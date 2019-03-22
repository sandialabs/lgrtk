#pragma once

#include <lgr_allocator.hpp>
#include <lgr_memory_pool.hpp>

namespace lgr {

template <class T>
concurrent_pooled_allocator<T>::concurrent_pooled_allocator(concurrent_memory_pool& pool_in) noexcept
  :m_pool(&pool_in)
{
}

template <class T>
bool concurrent_pooled_allocator<T>::operator==(concurrent_pooled_allocator const& other) noexcept
{
  return m_pool == other.m_pool;
}

template <class T>
bool concurrent_pooled_allocator<T>::operator!=(concurrent_pooled_allocator const& other) noexcept
{
  return m_pool != other.m_pool;
}

template <class T>
T* concurrent_pooled_allocator<T>::allocate(std::size_t n)
{
  return static_cast<T*>(m_pool->allocate(n * sizeof(T)));
}

template <class T>
void concurrent_pooled_allocator<T>::deallocate(T* p, std::size_t n)
{
  m_pool->deallocate(p, n * sizeof(T));
}

template <class T>
device_allocator<T>::device_allocator(device_memory_pool& pool_in) noexcept
  :concurrent_pooled_allocator<T>(pool_in)
{
}

template <class T>
pinned_allocator<T>::pinned_allocator(pinned_memory_pool& pool_in) noexcept
  :concurrent_pooled_allocator<T>(pool_in)
{
}

}
