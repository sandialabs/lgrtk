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
  constexpr inline Tag operator+(Tag const& other) const noexcept {
    return Tag(i + other.i);
  }
  constexpr inline Tag operator-(Tag const& other) const noexcept {
    return Tag(i - other.i);
  }
  inline Tag& operator++() noexcept {
    ++i;
    return *static_cast<Tag*>(this);
  }
  inline Tag operator++(int) const noexcept {
    auto const old = *this;
    ++i;
    return old;
  }
  inline Tag& operator--() noexcept {
    --i;
    return *static_cast<Tag*>(this);
  }
  inline Tag operator--(int) const noexcept {
    auto const old = *this;
    --i;
    return old;
  }
  constexpr inline bool operator==(Tag const& other) const noexcept {
    return i == other.i;
  }
  constexpr inline bool operator!=(Tag const& other) const noexcept {
    return i != other.i;
  }
  constexpr inline bool operator>(Tag const& other) const noexcept {
    return i > other.i;
  }
  constexpr inline bool operator<(Tag const& other) const noexcept {
    return i < other.i;
  }
  constexpr inline bool operator>=(Tag const& other) const noexcept {
    return i >= other.i;
  }
  constexpr inline bool operator<=(Tag const& other) const noexcept {
    return i <= other.i;
  }
  inline Tag& operator+=(Tag const& other) noexcept {
    i += other.i;
    return *static_cast<Tag*>(this);
  }
  inline Tag& operator-=(Tag const& other) noexcept {
    i -= other.i;
    return *static_cast<Tag*>(this);
  }
  explicit inline operator Integral() const noexcept {
    return i;
  }
  using size_t_std = std::size_t;
  explicit inline operator size_t_std() const noexcept {
    return size_t_std(i);
  }
  constexpr inline Tag operator*(Integral const n) const noexcept {
    return Tag(i * n);
  }
  constexpr inline Tag operator/(Integral const n) const noexcept {
    return Tag(i / n);
  }
};

}
