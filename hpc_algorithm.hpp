#pragma once

#include <hpc_macros.hpp>
#include <hpc_execution.hpp>
#include <hpc_range.hpp>
#include <hpc_functional.hpp>

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
HPC_ALWAYS_INLINE HPC_HOST_DEVICE void move(local_policy, InputRange const& input, OutputRange& output) noexcept
{
  auto first = input.begin();
  auto const last = input.end();
  auto d_first = output.begin();
  while (first != last) {
    *d_first++ = std::move(*first++);
  }
}

template <class InputRange, class OutputRange>
void move(serial_policy, InputRange const& input, OutputRange& output)
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

template <class Range, class T, class BinaryOp, class UnaryOp>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE T transform_reduce(
    local_policy, Range const& range, T init, BinaryOp binary_op, UnaryOp unary_op) noexcept {
  auto first = range.begin();
  auto const last = range.end();
  for (; first != last; ++first) {
    init = binary_op(std::move(init), unary_op(*first));
  }
  return init;
}

template <class Range, class T, class BinaryOp, class UnaryOp>
T transform_reduce(
    serial_policy, Range const& range, T init, BinaryOp binary_op, UnaryOp unary_op) {
  auto first = range.begin();
  auto const last = range.end();
  for (; first != last; ++first) {
    init = binary_op(std::move(init), unary_op(*first));
  }
  return init;
}

template <class Range, class T>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE T
reduce(local_policy policy, Range const& range, T init) noexcept {
  using input_value_type = typename Range::value_type;
  auto const unop = [] (input_value_type const i) { return T(i); };
  return transform_reduce(policy, range, init, plus<T>(), unop);
}

template <class Range, class T>
T reduce(serial_policy policy, Range const& range, T init) {
  using input_value_type = typename Range::value_type;
  auto const unop = [] (input_value_type const i) { return T(i); };
  return transform_reduce(policy, range, init, plus<T>(), unop);
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

template <class InputRange, class OutputRange, class BinaryOp, class UnaryOp>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE void
transform_inclusive_scan(local_policy, InputRange const& input, OutputRange& output, BinaryOp binary_op, UnaryOp unary_op) noexcept
{
  auto first = input.begin();
  auto last = input.end();
  auto d_first = output.begin();
  if (first == last) return;
  auto sum = unary_op(*first);
  *d_first = sum;
  while (++first != last) {
    sum = binary_op(std::move(sum), unary_op(*first));
    *(++d_first) = sum;
  }
}

template <class InputRange, class OutputRange, class BinaryOp, class UnaryOp>
void
transform_inclusive_scan(serial_policy, InputRange const& input, OutputRange& output, BinaryOp binary_op, UnaryOp unary_op) noexcept
{
  auto first = input.begin();
  auto last = input.end();
  auto d_first = output.begin();
  if (first == last) return;
  auto sum = unary_op(*first);
  *d_first = sum;
  while (++first != last) {
    sum = binary_op(std::move(sum), unary_op(*first));
    *(++d_first) = sum;
  }
}

}
