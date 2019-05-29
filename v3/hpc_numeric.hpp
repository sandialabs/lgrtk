#pragma once

#include <hpc_transform_reduce.hpp>

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
T reduce(serial_policy policy, Range const& range, T init) {
  using input_value_type = typename Range::value_type;
  auto const unop = [] (input_value_type const i) { return T(i); };
  return transform_reduce(policy, range, init, plus<T>(), unop);
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
