#pragma once

#include <cstddef>

#include <hpc_macros.hpp>

namespace hpc {

template <class Tag, class Integral = std::ptrdiff_t>
class index {
  Integral i;
public:
  using integral_type = Integral;
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr index(std::ptrdiff_t i_in) noexcept : i(integral_type(i_in)) {}
  HPC_ALWAYS_INLINE index() noexcept = default;
  HPC_ALWAYS_INLINE index(index const&) noexcept = default;
  HPC_ALWAYS_INLINE index(index&&) noexcept = default;
  HPC_ALWAYS_INLINE index& operator=(index const&) noexcept = default;
  HPC_ALWAYS_INLINE index& operator=(index&&) noexcept = default;
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr index operator+(index const& other) const noexcept {
    return index(i + other.i);
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr index operator-(index const& other) const noexcept {
    return index(i - other.i);
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE index& operator++() noexcept {
    ++i;
    return *this;
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE index operator++(int) const noexcept {
    auto const old = *this;
    ++i;
    return old;
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE index& operator--() noexcept {
    --i;
    return *static_cast<index*>(this);
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE index operator--(int) const noexcept {
    auto const old = *this;
    --i;
    return old;
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr bool operator==(index const& other) const noexcept {
    return i == other.i;
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr bool operator!=(index const& other) const noexcept {
    return i != other.i;
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr bool operator>(index const& other) const noexcept {
    return i > other.i;
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr bool operator<(index const& other) const noexcept {
    return i < other.i;
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr bool operator>=(index const& other) const noexcept {
    return i >= other.i;
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr bool operator<=(index const& other) const noexcept {
    return i <= other.i;
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE index& operator+=(index const& other) noexcept {
    i += other.i;
    return *static_cast<index*>(this);
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE index& operator-=(index const& other) noexcept {
    i -= other.i;
    return *static_cast<index*>(this);
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr integral_type get() const noexcept {
    return i;
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE explicit constexpr operator std::ptrdiff_t() const noexcept {
    return std::ptrdiff_t(i);
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE explicit constexpr operator std::size_t() const noexcept {
    return std::size_t(i);
  }
#ifdef HPC_CUDA
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE explicit constexpr operator int() const noexcept {
    return int(i);
  }
#endif
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr index operator*(integral_type const n) const noexcept {
    return index(i * n);
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr index operator/(integral_type const n) const noexcept {
    return index(i / n);
  }
  template <class Tag2, class Integral2>
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE explicit constexpr operator ::hpc::index<Tag2, Integral2>() const noexcept {
    return i;
  }
};

template <class L, class R>
class product_tag {};

template <class L, class R, class LI, class RI>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr index<product_tag<L, R>, decltype(LI() * RI())>
operator*(index<L, LI> left, index<R, RI> right) noexcept {
  return left.get() * right.get();
}

template <class L, class R, class LI, class RI>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr index<L, decltype(LI() / RI())>
operator/(index<product_tag<L, R>, LI> left, index<R, RI> right) noexcept {
  return left.get() / right.get();
}

}
