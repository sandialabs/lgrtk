#pragma once

#include <iterator>

#include <lgr_functional.hpp>
#include <lgr_fill.hpp>

namespace lgr {

template<class InputRange, class OutputRange, class BinaryOp, class UnaryOp>
auto transform_inclusive_scan(InputRange&& input, OutputRange&& output, BinaryOp binary_op, UnaryOp unary_op)
{
  auto first = input.begin();
  auto last = input.end();
  auto d_first = output.begin();
  if (first == last) return d_first;
  auto sum = unary_op(*first);
  *d_first = sum;
  while (++first != last) {
    sum = binary_op(std::move(sum), unary_op(*first));
    *(++d_first) = sum;
  }
  return ++d_first;
}

template <class InputRange, class OutputRange>
void offset_scan(InputRange const& input, OutputRange& output) {
  auto it = output.begin();
  auto const first = it;
  ++it;
  auto const second = it;
  auto first_range = iterator_range<decltype(it)>(first, second);
  using input_value_type = typename InputRange::value_type;
  using output_value_type = typename OutputRange::value_type;
  fill(first_range, output_value_type(0));
  auto const end = output.end();
  auto rest = iterator_range<decltype(it)>(second, end);
  auto const unop = [] (input_value_type const i) { return output_value_type(i); };
  transform_inclusive_scan(input, rest,
      plus<output_value_type>(), unop);
}

}
