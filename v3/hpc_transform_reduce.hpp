#pragma once

#include <hpc_execution.hpp>

#ifdef HPC_CUDA

#include <thrust/execution_policy.h>
#include <thrust/transform_reduce.h>

#endif

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
HPC_NOINLINE T
transform_reduce(
    serial_policy, Range const& range, T init, BinaryOp binary_op, UnaryOp unary_op) {
  auto first = range.begin();
  auto const last = range.end();
  for (; first != last; ++first) {
    init = binary_op(std::move(init), unary_op(*first));
  }
  return init;
}

#ifdef HPC_CUDA

namespace impl {

template <class Iterator, class T, class BinaryOp, class UnaryOp>
T
transform_reduce(
    cuda_policy, Iterator first, Iterator last, T init, BinaryOp binary_op, UnaryOp unary_op) {
  return ::thrust::transform_reduce(::thrust::device, first, last, unary_op, init, binary_op);
}

template <class Index, class T, class BinaryOp, class UnaryOp>
T
transform_reduce(
    cuda_policy,
    ::hpc::counting_iterator<Index> first,
    ::hpc::counting_iterator<Index> last,
    T init,
    BinaryOp binary_op,
    UnaryOp unary_op) {
  int const n = int(last - first);
  thrust::counting_iterator<int> new_first(0);
  thrust::counting_iterator<int> new_last(n);
  return ::thrust::transform_reduce(::thrust::device, new_first, new_last, unary_op, init, binary_op);
}

template <class TStored, class TResult, class Index, class BinaryOp, class UnaryOp>
TResult
transform_reduce(
    cuda_policy,
    ::hpc::pointer_iterator<TStored, Index> first,
    ::hpc::pointer_iterator<TStored, Index> last,
    TResult init,
    BinaryOp binary_op,
    UnaryOp unary_op) {
  TStored* const new_first = &(*first);
  TStored* const new_last = &(*last);
  return ::thrust::transform_reduce(::thrust::device, new_first, new_last, unary_op, init, binary_op);
}

}

template <class Range, class T, class BinaryOp, class UnaryOp>
HPC_NOINLINE T
transform_reduce(
    cuda_policy policy, Range const& range, T init, BinaryOp binary_op, UnaryOp unary_op) {
  return ::hpc::impl::transform_reduce(policy, range.begin(), range.end(), init, binary_op, unary_op);
}

#endif

}
