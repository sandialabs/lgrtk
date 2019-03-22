#pragma once

#include <iterator>

namespace lgr {

template<class InputIt, class OutputIt>
OutputIt inclusive_scan(InputIt first, InputIt last, 
                     OutputIt d_first)
{
	if (first == last) return d_first;
	typename std::iterator_traits<InputIt>::value_type sum = *first;
	*d_first = sum;
	while (++first != last) {
		sum = std::move(sum) + *first;
		*(++d_first) = sum;
	}
	return ++d_first;
}

}
