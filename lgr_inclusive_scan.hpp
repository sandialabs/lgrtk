#pragma once

#include <iterator>

namespace lgr {

template<class InputRange, class OutputRange>
void inclusive_scan(InputRange&& input, OutputRange&& output)
{
  auto first = input.begin();
  auto last = input.end();
  auto d_first = output.begin();
	if (first == last) return;
	typename std::iterator_traits<decltype(first)>::value_type sum = *first;
	*d_first = sum;
	while (++first != last) {
		sum = std::move(sum) + *first;
		*(++d_first) = sum;
	}
	return;
}

}
