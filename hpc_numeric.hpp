#pragma once

#include <hpc_transform_reduce.hpp>

#ifdef HPC_CUDA
#include <thrust/execution_policy.h>
#include <thrust/transform_scan.h>
#endif

namespace hpc {

namespace impl {

template <class T>
class pi;
template <>
class pi<float> { public: static constexpr float value   = 3.14159265f; };
template <>
class pi<double> { public: static constexpr double value = 3.141592653589793238; };

}

template <class T>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr T pi() noexcept { return ::hpc::impl::pi<T>::value; }

template <class Range, class T>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE T
reduce(local_policy policy, Range const& range, T init) noexcept {
  using input_value_type = typename Range::value_type;
  auto const unop = [] (input_value_type const i) { return T(i); };
  return transform_reduce(policy, range, init, plus<T>(), unop);
}

template <class Range, class T>
HPC_NOINLINE T reduce(serial_policy policy, Range const& range, T init) {
  using input_value_type = typename Range::value_type;
  auto const unop = [] (input_value_type const i) { return T(i); };
  return transform_reduce(policy, range, init, plus<T>(), unop);
}

#ifdef HPC_CUDA

template <class Range, class T>
HPC_NOINLINE T reduce(cuda_policy policy, Range const& range, T init) {
  using input_value_type = typename Range::value_type;
  auto const unop = [] HPC_DEVICE (input_value_type const i) { return T(i); };
  return hpc::transform_reduce(policy, range, init, hpc::plus<T>(), unop);
}

#endif

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
HPC_NOINLINE void
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

#ifdef HPC_CUDA

namespace impl {

template <class InputIterator, class OutputIterator, class BinaryOp, class UnaryOp>
HPC_NOINLINE void
transform_inclusive_scan(cuda_policy, InputIterator first, InputIterator last, OutputIterator d_first, BinaryOp binary_op, UnaryOp unary_op) noexcept
{
  thrust::transform_inclusive_scan(thrust::device, first, last, d_first, unary_op, binary_op);
}

template <class T, class Index, class UnaryOp>
HPC_NOINLINE void
transform_inclusive_scan(cuda_policy,
    ::hpc::counting_iterator<Index> first,
    ::hpc::counting_iterator<Index> last,
    ::hpc::pointer_iterator<T, Index> d_first,
    ::hpc::plus<T>,
    UnaryOp unary_op) noexcept
{
  ::hpc::counting_iterator<Index> old_zero(0);
  ::thrust::counting_iterator<int> new_first(int(first - old_zero));
  ::thrust::counting_iterator<int> new_last(int(last - old_zero));
  T* new_d_first = &(*d_first);
  thrust::transform_inclusive_scan(thrust::device, new_first, new_last, new_d_first, unary_op, thrust::plus<T>());
}

template <class TStored, class TResult, class Index, class UnaryOp>
HPC_NOINLINE void
transform_inclusive_scan(cuda_policy,
    ::hpc::pointer_iterator<TStored, Index> first,
    ::hpc::pointer_iterator<TStored, Index> last,
    ::hpc::pointer_iterator<TResult, Index> d_first,
    ::hpc::plus<TResult>,
    UnaryOp unary_op) noexcept
{
  TStored* const new_first = &(*first);
  TStored* const new_last = &(*last);
  TResult* const new_d_first = &(*d_first);
  thrust::transform_inclusive_scan(thrust::device, new_first, new_last, new_d_first, unary_op, thrust::plus<T>());
}

}

template <class InputRange, class OutputRange, class BinaryOp, class UnaryOp>
HPC_NOINLINE void
transform_inclusive_scan(cuda_policy policy, InputRange const& input, OutputRange& output, BinaryOp binary_op, UnaryOp unary_op) noexcept
{
  ::hpc::impl::transform_inclusive_scan(policy, input.begin(), input.end(), output.begin(), binary_op, unary_op);
}

#endif

template <class ExecutionPolicy, class InputRange, class OutputRange>
void offset_scan(ExecutionPolicy policy, InputRange const& input, OutputRange& output) {
  auto it = output.begin();
  auto const first = it;
  ++it;
  auto const second = it;
  auto const first_range = ::hpc::iterator_range<decltype(it)>(first, second);
  using input_value_type = typename InputRange::value_type;
  using output_value_type = typename OutputRange::value_type;
  ::hpc::fill(policy, first_range, output_value_type(0));
  auto const end = output.end();
  auto const rest = iterator_range<decltype(it)>(second, end);
  ::hpc::transform_inclusive_scan(policy, input, rest, ::hpc::plus<output_value_type>(),
      ::hpc::cast<output_value_type, input_value_type>());
}

}
