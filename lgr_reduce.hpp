#pragma once

#include <lgr_functional.hpp>

namespace lgr {

template <class Range, class T, class BinaryOp, class UnaryOp>
T transform_reduce(Range&& range, T init, BinaryOp binary_op, UnaryOp unary_op) {
  auto first = range.begin();
  auto const last = range.end();
  for (; first != last; ++first) {
    init = binary_op(std::move(init), unary_op(*first));
  }
  return init;
}

template <class Range, class T>
T reduce(Range&& range, T init) {
  return transform_reduce(range, init, plus<T>(), identity<T>());
}

}
