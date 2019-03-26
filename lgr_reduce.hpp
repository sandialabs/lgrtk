#pragma once

#include <lgr_functional.hpp>

namespace lgr {

template <class Range, class T, class BinaryOp>
T reduce(Range&& range, T init, BinaryOp binary_op) {
  auto first = range.begin();
  auto const last = range.end();
  for (; first != last; ++first) {
    init = binary_op(std::move(init), *first);
  }
  return init;
}

}
