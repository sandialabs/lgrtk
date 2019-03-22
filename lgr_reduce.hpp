#pragma once

#include <lgr_functional.hpp>

namespace lgr {

template <class ForwardIt, class T, class BinaryOp>
T reduce(
    ForwardIt first, ForwardIt last,
    T init, BinaryOp binary_op) {
  for (; first != last; ++first) {
    init = binary_op(std::move(init), *first);
  }
  return init;
}

}
