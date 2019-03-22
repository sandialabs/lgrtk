#pragma once

#include <lgr_product_range.hpp>
#include <lgr_int_range.hpp>

namespace lgr {

inline product_range<counting_iterator<int>, AOS, int, int>
operator*(int_range const& a, int_range const& b) {
  return product_range<counting_iterator<int>, AOS, int, int>(
      counting_iterator<int>(0),
      a.size(), b.size());
}

}
