#pragma once

#include <hpc_macros.hpp>
#include <hpc_execution.hpp>
#include <hpc_range.hpp>
#include <hpc_functional.hpp>
#include <hpc_transform_reduce.hpp>

#ifdef HPC_CUDA
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/fill.h>
#endif

namespace hpc {

template <class T>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE void swap(T& a, T& b) noexcept {
  T c(std::move(a));
  a = std::move(b);
  b = std::move(c);
};

template <class Range, class UnaryFunction>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE void for_each(local_policy, Range&& r, UnaryFunction f) noexcept {
  for (auto it = r.begin(), end = r.end(); it != end; ++it) {
    f(*it);
  }
}

template <class Range, class UnaryFunction>
void for_each(serial_policy, Range&& r, UnaryFunction f) {
  for (auto it = r.begin(), end = r.end(); it != end; ++it) {
    f(*it);
  }
}

#ifdef HPC_CUDA
template <class Range, class UnaryFunction>
void for_each(cuda_policy, Range&& r, UnaryFunction f) {
  thrust::for_each(thrust::device, r.begin(), r.end(), f);
}
#endif

template <class FromRange, class ToRange>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE void copy(local_policy, FromRange const& from, ToRange& to) noexcept
{
  auto first = from.begin();
  auto const last = from.end();
  auto d_first = to.begin();
  while (first != last) {
    *d_first++ = *first++;
  }
}

template <class FromRange, class ToRange>
HPC_NOINLINE void copy(serial_policy, FromRange const& from, ToRange& to) {
  auto first = from.begin();
  auto const last = from.end();
  auto d_first = to.begin();
  while (first != last) {
    *d_first++ = *first++;
  }
}

#ifdef HPC_CUDA

template <class FromRange, class ToRange>
HPC_NOINLINE void copy(cuda_policy, FromRange const& from, ToRange& to) {
  auto const first = from.begin();
  auto const d_first = to.begin();
  auto functor = [=] HPC_DEVICE (typename FromRange::size_type const i) {
    d_first[i] = FromRange::value_type(first[i]);
  };
  int n = int(from.size());
  thrust::counting_iterator<int> new_first(0);
  thrust::counting_iterator<int> new_last(n);
  thrust::for_each(thrust::device, new_first, new_last, functor);
}

#endif

template <class InputRange, class OutputRange>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE void move(local_policy, InputRange& input, OutputRange& output) noexcept
{
  auto first = input.begin();
  auto const last = input.end();
  auto d_first = output.begin();
  while (first != last) {
    *d_first++ = std::move(*first++);
  }
}

template <class InputRange, class OutputRange>
HPC_NOINLINE void move(serial_policy, InputRange& input, OutputRange& output)
{
  auto first = input.begin();
  auto const last = input.end();
  auto d_first = output.begin();
  while (first != last) {
    *d_first++ = std::move(*first++);
  }
}

#ifdef HPC_CUDA

template <class InputRange, class OutputRange>
HPC_NOINLINE void move(cuda_policy policy, InputRange& input, OutputRange& output)
{
  using size_type = typename InputRange::size_type;
  auto const input_begin = input.begin();
  auto const output_begin = output.begin();
  auto functor = [=] HPC_DEVICE (size_type const i) {
    output_begin[i] = std::move(input_begin[i]);
  };
  auto size = input.size();
  auto range = ::hpc::counting_range<size_type>(size);
  ::hpc::for_each(policy, range, functor);
}

#endif

template <class Range, class T>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE void fill(local_policy, Range& r, T value) noexcept {
  auto first = r.begin();
  auto const last = r.end();
  for (; first != last; ++first) {
    *first = value;
  }
}

template <class Range, class T>
HPC_NOINLINE void fill(serial_policy, Range& r, T value) {
  auto first = r.begin();
  auto const last = r.end();
  for (; first != last; ++first) {
    *first = value;
  }
}

#ifdef HPC_CUDA
template <class Range, class T>
HPC_NOINLINE void fill(cuda_policy, Range& r, T value) {
  thrust::fill(thrust::device, r.begin(), r.end(), value);
}
#endif

template <class Range, class UnaryPredicate>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE
bool any_of(local_policy policy, Range const& range, UnaryPredicate p) noexcept {
  return transform_reduce(policy, range, false, logical_or(), p);
}

template <class Range, class UnaryPredicate>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE
bool all_of(local_policy policy, Range const& range, UnaryPredicate p) noexcept {
  return transform_reduce(policy, range, true, logical_and(), p); 
}

template <class Range>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE
bool all_of(local_policy policy, Range const& range) noexcept {
  return all_of(policy, range, identity<bool>());
}

template <class Range, class UnaryPredicate>
bool any_of(serial_policy policy, Range const& range, UnaryPredicate p) {
  return transform_reduce(policy, range, false, logical_or(), p);
}

template <class Range, class UnaryPredicate>
bool all_of(serial_policy policy, Range const& range, UnaryPredicate p) {
  return transform_reduce(policy, range, true, logical_and(), p); 
}

template <class Range>
bool all_of(serial_policy policy, Range const& range) {
  return all_of(policy, range, identity<bool>());
}

template <class Range>
bool any_of(serial_policy policy, Range const& range) {
  return any_of(policy, range, identity<bool>());
}

}
