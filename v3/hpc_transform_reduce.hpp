#pragma once

namespace hpc {

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

}
