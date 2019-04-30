#pragma once

#include <cstddef>

namespace lgr {

template <class Integral, class Derived>
class index {
  Integral i;
public:
  constexpr explicit inline index(Integral const& i_in) noexcept : i(i_in) {}
  inline index() noexcept = default;
  inline index(index const&) noexcept = default;
  inline index(index&&) noexcept = default;
  inline index& operator=(index const&) noexcept = default;
  inline index& operator=(index&&) noexcept = default;
  constexpr inline Derived operator+(Derived const& other) const noexcept {
    return Derived(i + other.i);
  }
  constexpr inline Derived operator-(Derived const& other) const noexcept {
    return Derived(i - other.i);
  }
  inline Derived& operator++() noexcept {
    ++i;
    return *static_cast<Derived*>(this);
  }
  inline Derived operator++(int) const noexcept {
    auto const old = *this;
    ++i;
    return old;
  }
  inline Derived& operator--() noexcept {
    --i;
    return *static_cast<Derived*>(this);
  }
  inline Derived operator--(int) const noexcept {
    auto const old = *this;
    --i;
    return old;
  }
  constexpr inline bool operator==(Derived const& other) const noexcept {
    return i == other.i;
  }
  constexpr inline bool operator!=(Derived const& other) const noexcept {
    return i != other.i;
  }
  constexpr inline bool operator>(Derived const& other) const noexcept {
    return i > other.i;
  }
  constexpr inline bool operator<(Derived const& other) const noexcept {
    return i < other.i;
  }
  constexpr inline bool operator>=(Derived const& other) const noexcept {
    return i >= other.i;
  }
  constexpr inline bool operator<=(Derived const& other) const noexcept {
    return i <= other.i;
  }
  inline Derived& operator+=(Derived const& other) noexcept {
    i += other.i;
    return *static_cast<Derived*>(this);
  }
  inline Derived& operator-=(Derived const& other) noexcept {
    i -= other.i;
    return *static_cast<Derived*>(this);
  }
  explicit constexpr inline operator Integral() const noexcept {
    return i;
  }
  using size_t_std = std::size_t;
  explicit inline operator size_t_std() const noexcept {
    return size_t_std(i);
  }
  constexpr inline Derived operator*(Integral const n) const noexcept {
    return Derived(i * n);
  }
  constexpr inline Derived operator/(Integral const n) const noexcept {
    return Derived(i / n);
  }
};

}
