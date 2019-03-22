#pragma once

#include <lgr_for_each.hpp>
#include <lgr_counting_range.hpp>

namespace lgr {

template <class LeftRange, class RightRange>
void copy(LeftRange const& left, RightRange& right) {
  auto const left_iterator = left.begin();
  using difference_type = typename decltype(left_iterator)::difference_type;
  difference_type const n = left.size();
  auto const right_iterator = right.begin();
  auto functor = [=] (difference_type const i) {
    right_iterator[i] = left_iterator[i];
  };
  lgr::for_each(counting_range<difference_type>(n), functor);
}

}

