#pragma once

namespace lgr {

template <class Integral, class Tag>
class index {
  Integral i;
public:
  constexpr explicit inline index(Integral const& i_in) noexcept : i(i_in) {}
  inline index(index const&) noexcept = default;
  inline index(index&&) noexcept = default;
  inline index& operator=(index const&) noexcept = default;
  inline index& operator=(index&&) noexcept = default;
  constexpr inline index operator+(Integral const n) const noexcept {
    return index(i + n);
  }
  constexpr inline Integral operator-(index const& other) const noexcept {
    return i - other.i;
  }
  inline index& operator++() noexcept {
    ++i;
    return *this;
  }
  inline index operator++(int) const noexcept {
    auto const old = *this;
    ++i;
    return old;
  }
  inline index& operator--() noexcept {
    --i;
    return *this;
  }
  inline index operator--(int) const noexcept {
    auto const old = *this;
    --i;
    return old;
  }
  constexpr inline bool operator==(index const& other) const noexcept {
    return i == other.i;
  }
  constexpr inline bool operator!=(index const& other) const noexcept {
    return i != other.i;
  }
  constexpr inline bool operator>(index const& other) const noexcept {
    return i > other.i;
  }
  constexpr inline bool operator<(index const& other) const noexcept {
    return i < other.i;
  }
  constexpr inline bool operator>=(index const& other) const noexcept {
    return i >= other.i;
  }
  constexpr inline bool operator<=(index const& other) const noexcept {
    return i <= other.i;
  }
  inline index& operator+=(Integral const n) noexcept {
    i += n;
    return *this;
  }
  inline index& operator-=(Integral const n) noexcept {
    i -= n;
    return *this;
  }
};

}
