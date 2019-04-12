#pragma once

#include <iterator>

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

template<class InputRange, class OutputRange, class T, class BinaryOp, class UnaryOp>
auto transform_exclusive_scan(InputRange&& input, OutputRange&& output, T init, BinaryOp binary_op, UnaryOp unary_op)
{
  auto first = input.begin();
  auto last = input.end();
  auto d_first = output.begin();
	*d_first = init;
	while (++first != last) {
		init = binary_op(std::move(init), unary_op(*first));
		*(++d_first) = init;
	}
	return ++d_first;
}

}
