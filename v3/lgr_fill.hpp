#pragma once

#include <lgr_for_each.hpp>

namespace lgr {

template <class Range, class T>
void fill(Range& r, T const value) {
  auto functor = [=] (typename Range::reference ref) {
    ref = value;
  };
  lgr::for_each(r, functor);
}

}
