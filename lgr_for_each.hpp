#pragma once

namespace lgr {

template <class Range, class UnaryFunction>
void for_each(Range const& r, UnaryFunction f) {
  for (auto it = r.begin(), end = r.end(); it != end; ++it) {
    f(*it);
  }
}

template <class Range, class UnaryFunction>
void for_each(Range& r, UnaryFunction f) {
  for (auto it = r.begin(), end = r.end(); it != end; ++it) {
    f(*it);
  }
}

}
