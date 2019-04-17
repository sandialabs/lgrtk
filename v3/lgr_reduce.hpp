#pragma once

#include <lgr_functional.hpp>

namespace lgr {

template <class Range, class T, class BinaryOp, class UnaryOp>
T transform_reduce(Range const& range, T init, BinaryOp binary_op, UnaryOp unary_op) {
  auto first = range.begin();
  auto const last = range.end();
  for (; first != last; ++first) {
    init = binary_op(std::move(init), unary_op(*first));
  }
  return init;
}

template <class Range, class T>
T reduce(Range const& range, T init) {
  using input_value_type = typename Range::value_type;
  auto const unop = [] (input_value_type const i) { return T(i); };
  return transform_reduce(range, init, plus<T>(), unop);
}

}
