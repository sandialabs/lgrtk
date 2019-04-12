#pragma once

#include <iterator>
#include <lgr_functional.hpp>

namespace lgr {

template<class InputRange, class OutputRange, class BinaryOp, class UnaryOp>
void transform_inclusive_scan(InputRange&& input, OutputRange&& output, BinaryOp binary_op, UnaryOp unary_op)
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
  return;
}

template <class InputRange, class OutputRange>
void offset_scan(InputRange&& input, OutputRange&& output)
{
  using input_type = typename std::decay<InputRange>::type;
  using output_type = typename std::decay<OutputRange>::type;
  using input_value_type = typename input_type::value_type;
  using output_value_type = typename output_type::value_type;
#ifndef NDEBUG
  using size_type = typename InputRange::size_type;
#endif
  assert(output.size() == input.size() + size_type(1));
  auto iterator = output.begin();
  *iterator = output_value_type(0);
  ++iterator;
  auto const rest = iterator_range<decltype(iterator)>(iterator, output.end());
  plus<output_value_type> const binop;
  auto const unop = [] (input_value_type const i) -> output_value_type { return output_value_type(i); };
  transform_inclusive_scan(input, rest, binop, unop);
}

}
