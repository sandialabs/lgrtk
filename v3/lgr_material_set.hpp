#pragma once

#include <lgr_mesh_indices.hpp>

namespace lgr {

HPC_ALWAYS_INLINE HPC_HOST_DEVICE int popcount(unsigned x) noexcept {
#ifdef __CUDA_ARCH__
  return __popc(x);
#else
  return __builtin_popcount(x);
#endif
}

HPC_ALWAYS_INLINE HPC_HOST_DEVICE int popcount(unsigned long x) noexcept {
#ifdef __CUDA_ARCH__
  return __popcll(x);
#else
  return __builtin_popcountl(x);
#endif
}

HPC_ALWAYS_INLINE HPC_HOST_DEVICE int popcount(unsigned long long x) noexcept {
#ifdef __CUDA_ARCH__
  return __popcll(x);
#else
  return __builtin_popcountll(x);
#endif
}

class material_set {
  std::uint64_t bits;
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE explicit constexpr material_set(std::uint64_t const bits_in) noexcept : bits(bits_in) {}
public:
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE explicit constexpr material_set(material_index const material) noexcept : bits(std::uint64_t(1) << material.get()) {}
  HPC_ALWAYS_INLINE material_set() noexcept = default;
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr material_set operator|(material_set const other) const noexcept {
    return material_set(bits | other.bits);
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr bool contains(material_set const other) const noexcept {
    return (bits | other.bits) == bits;
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr explicit operator std::uint64_t() const noexcept { return bits; }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE int size() const noexcept { return popcount(bits); }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr material_set operator-(material_set const other) const noexcept {
    return material_set(bits & (~other.bits));
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr static material_set none() noexcept { return material_set(std::uint64_t(0)); }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr static material_set all(material_index const size) noexcept {
    return material_set((std::uint64_t(1) << size.get()) - 1);
  }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr bool operator==(material_set const other) const noexcept { return bits == other.bits; }
};

}
