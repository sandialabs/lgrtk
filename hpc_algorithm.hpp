#pragma once

#include <hpc_macros.hpp>
#include <hpc_execution.hpp>
#include <hpc_range.hpp>
#include <hpc_functional.hpp>
#include <hpc_transform_reduce.hpp>

namespace hpc {

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
void copy(serial_policy, FromRange const& from, ToRange& to) {
  auto first = from.begin();
  auto const last = from.end();
  auto d_first = to.begin();
  while (first != last) {
    *d_first++ = *first++;
  }
}

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
void move(serial_policy, InputRange& input, OutputRange& output)
{
  auto first = input.begin();
  auto const last = input.end();
  auto d_first = output.begin();
  while (first != last) {
    *d_first++ = std::move(*first++);
  }
}

template <class Range, class T>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE void fill(local_policy, Range& r, T value) noexcept {
  auto first = r.begin();
  auto const last = r.end();
  for (; first != last; ++first) {
    *first = value;
  }
}

template <class Range, class T>
void fill(serial_policy, Range& r, T value) {
  auto first = r.begin();
  auto const last = r.end();
  for (; first != last; ++first) {
    *first = value;
  }
}

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

}
