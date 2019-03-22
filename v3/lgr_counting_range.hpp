#pragma once

#include <lgr_iterator_range.hpp>
#include <lgr_counting_iterator.hpp>

namespace lgr {

template <class T>
class counting_range : public iterator_range<counting_iterator<T>> {
  public:
  explicit inline counting_range(T const& first, T const& last):
  iterator_range<counting_iterator<T>>(counting_iterator<T>(first), counting_iterator<T>(last))
  {
  }
  explicit inline counting_range(T const& last):
  iterator_range<counting_iterator<T>>(counting_iterator<T>(T(0)), counting_iterator<T>(last))
  {
  }
};

}
