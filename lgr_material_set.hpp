#pragma once

#include <lgr_mesh_indices.hpp>

namespace lgr {

inline int popcount(unsigned x) noexcept { return __builtin_popcount(x); }
inline int popcount(unsigned long x) noexcept { return __builtin_popcountl(x); }
inline int popcount(unsigned long long x) noexcept { return __builtin_popcountll(x); }

class material_set {
  std::uint64_t bits;
  explicit constexpr inline material_set(std::uint64_t const bits_in) noexcept : bits(bits_in) {}
public:
  explicit constexpr inline material_set(material_index const material) noexcept : bits(std::uint64_t(1) << material.get()) {}
  inline material_set() noexcept = default;
  constexpr inline material_set operator|(material_set const other) const noexcept {
    return material_set(bits | other.bits);
  }
  constexpr inline bool contains(material_set const other) const noexcept {
    return (bits | other.bits) == bits;
  }
  constexpr explicit inline operator std::uint64_t() const noexcept { return bits; }
  inline int size() const noexcept { return popcount(bits); }
  constexpr inline material_set operator-(material_set const other) const noexcept {
    return material_set(bits & (~other.bits));
  }
  constexpr static inline material_set none() noexcept { return material_set(std::uint64_t(0)); }
  constexpr static inline material_set all(material_index const size) noexcept {
    return material_set((std::uint64_t(1) << size.get()) - 1);
  }
  constexpr inline bool operator==(material_set const other) const noexcept { return bits == other.bits; }
};

}
