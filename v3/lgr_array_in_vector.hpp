#pragma once

#include <lgr_layout.hpp>
#include <lgr_vector.hpp>
#include <lgr_range_product.hpp>

namespace lgr {

template <class T, int N, layout L, class OuterIndex, class InnerIndex = int>
using array_in_vector = iterator_range<inner_iterator<vector_iterator<T, OuterIndex>, L, OuterIndex, InnerIndex>>;

}
