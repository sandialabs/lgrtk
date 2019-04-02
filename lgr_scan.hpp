#pragma once

#include <iterator>

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

}
