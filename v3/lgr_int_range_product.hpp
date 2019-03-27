#pragma once

#include <lgr_product_range.hpp>
#include <lgr_counting_range.hpp>

namespace lgr {

static constexpr layout product_layout = AOS;

template <class OuterIndex, class InnerIndex>
inline product_range<
  counting_iterator<decltype(std::declval<OuterIndex>() * std::declval<InnerIndex>())>,
  product_layout,
  OuterIndex,
  InnerIndex>
operator*(counting_range<OuterIndex> const& a, counting_range<InnerIndex> const& b) {
  using ProductIndex = decltype(std::declval<OuterIndex>() * std::declval<InnerIndex>());
  using ProductIterator = counting_iterator<ProductIndex>;
  return product_range<
    ProductIterator,
    product_layout,
    OuterIndex,
    InnerIndex>(
      ProductIterator(ProductIndex(0)),
      a.size(), b.size());
}

}
