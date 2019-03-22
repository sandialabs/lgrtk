#pragma once

namespace lgr {

template <class T>
inline constexpr T max(T const& a, T const& b) noexcept {
  return (a < b) ? b : a;
}

template<class T>
inline constexpr T min(T const& a, T const& b) noexcept {
  return (b < a) ? b : a;
}

}
