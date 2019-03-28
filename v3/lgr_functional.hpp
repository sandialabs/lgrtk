#pragma once

#include <lgr_binary_ops.hpp>

namespace lgr {

template <class T>
struct minimum {
  inline constexpr T operator()(T const& a, T const& b) noexcept {
    return ::lgr::min(a, b);
  }
};

template <class T>
struct plus {
  inline constexpr T operator()(T const& a, T const& b) noexcept {
    return a + b;
  }
};

template <class T>
struct identity {
  inline constexpr T operator()(T const& a) noexcept {
    return a;
  }
};

}
