#pragma once

#include <iostream>
#include <cstdlib>

#include <lgr_print.hpp>

namespace lgr {

template <class ...types>
[[noreturn]] void fail(types... arguments) {
  print(std::cerr, arguments...);
  std::abort();
}

}
