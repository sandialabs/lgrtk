#pragma once

#include <hpc_vector3.hpp>
#include <hpc_matrix3x3.hpp>
#include <hpc_symmetric3x3.hpp>

namespace hpc {

template
  <int LengthPower
  ,int MassPower
  ,int TimePower
  ,int CurrentPower
  ,int TemperaturePower
  ,int AmountPower
  ,int IntensityPower
  >
class dimension {
  public:
  static constexpr int length_power = LengthPower;
  static constexpr int mass_power = MassPower;
  static constexpr int time_power = TimePower;
  static constexpr int current_power = CurrentPower;
  static constexpr int temperature_power = TemperaturePower;
  static constexpr int amount_power = AmountPower;
  static constexpr int intensity_power = IntensityPower;
};

template <class left, class right>
class multiply_dimensions {
  public:
    using type = dimension<
      left::length_power + right::length_power,
      left::mass_power + right::mass_power,
      left::time_power + right::time_power,
      left::current_power + right::current_power,
      left::temperature_power + right::temperature_power,
      left::amount_power + right::amount_power,
      left::intensity_power + right::intensity_power>;
};

template <class left, class right>
using multiply_dimensions_t = typename multiply_dimensions<left, right>::type;

template <class left, class right>
class divide_dimensions {
  public:
    using type = dimension<
      left::length_power - right::length_power,
      left::mass_power - right::mass_power,
      left::time_power - right::time_power,
      left::current_power - right::current_power,
      left::temperature_power - right::temperature_power,
      left::amount_power - right::amount_power,
      left::intensity_power - right::intensity_power>;
};

template <class left, class right>
using divide_dimensions_t = typename divide_dimensions<left, right>::type;

template <class dim, int factor>
class root_dimension {
  static_assert(dim::length_power % factor == 0, "length power not evenly divisible");
  static_assert(dim::mass_power % factor == 0, "mass power not evenly divisible");
  static_assert(dim::time_power % factor == 0, "time power not evenly divisible");
  static_assert(dim::current_power % factor == 0, "current power not evenly divisible");
  static_assert(dim::temperature_power % factor == 0, "temperature power not evenly divisible");
  static_assert(dim::amount_power % factor == 0, "amount power not evenly divisible");
  static_assert(dim::intensity_power % factor == 0, "intensity power not evenly divisible");
  public:
    using type = dimension<
      dim::length_power / factor,
      dim::mass_power / factor,
      dim::time_power / factor,
      dim::current_power / factor,
      dim::temperature_power / factor,
      dim::amount_power / factor,
      dim::intensity_power / factor>;
};

template <class dim, int factor>
using root_dimension_t = typename root_dimension<dim, factor>::type;

template <class dim, int factor>
class raise_dimension {
  public:
    using type = dimension<
      dim::length_power * factor,
      dim::mass_power * factor,
      dim::time_power * factor,
      dim::current_power * factor,
      dim::temperature_power * factor,
      dim::amount_power * factor,
      dim::intensity_power * factor>;
};

template <class dim, int factor>
using raise_dimension_t = typename raise_dimension<dim, factor>::type;

using no_dimension = dimension<0, 0, 0, 0, 0, 0, 0>;
using length_dimension = dimension<1, 0, 0, 0, 0, 0, 0>;
using mass_dimension = dimension<0, 1, 0, 0, 0, 0, 0>;
using time_dimension = dimension<0, 0, 1, 0, 0, 0, 0>;
using current_dimension = dimension<0, 0, 0, 1, 0, 0, 0>;
using temperature_dimension = dimension<0, 0, 0, 0, 1, 0, 0>;
using amount_dimension = dimension<0, 0, 0, 0, 0, 1, 0>;
using intensity_dimension = dimension<0, 0, 0, 0, 0, 0, 1>;
using speed_dimesion = divide_dimensions_t<length_dimension, time_dimension>;
using velocity_dimension = speed_dimesion;
using acceleration_dimension = divide_dimensions_t<speed_dimesion, time_dimension>;
using force_dimension = multiply_dimensions_t<mass_dimension, acceleration_dimension>;
using area_dimension = multiply_dimensions_t<length_dimension, length_dimension>;
using pressure_dimension = divide_dimensions_t<force_dimension, area_dimension>;
using volume_dimension = multiply_dimensions_t<area_dimension, length_dimension>;
using gradient_dimension = divide_dimensions_t<no_dimension, length_dimension>;
using density_dimension = divide_dimensions_t<mass_dimension, volume_dimension>;
using energy_dimension = multiply_dimensions_t<force_dimension, length_dimension>;
using power_dimension = divide_dimensions_t<energy_dimension, time_dimension>;
using specific_energy_dimension = divide_dimensions_t<energy_dimension, mass_dimension>;
using specific_energy_rate_dimension = divide_dimensions_t<specific_energy_dimension, time_dimension>;
using velocity_gradient_dimension = divide_dimensions_t<velocity_dimension, length_dimension>;
using dynamic_viscosity_dimension = divide_dimensions_t<pressure_dimension, velocity_gradient_dimension>;
using kinematic_viscosity_dimension = divide_dimensions_t<area_dimension, time_dimension>;
using power_dimension = divide_dimensions_t<energy_dimension, time_dimension>;
using heat_flux_dimension = divide_dimensions_t<power_dimension, area_dimension>;
using pressure_rate_dimension = divide_dimensions_t<pressure_dimension, time_dimension>;
using energy_density_dimension = divide_dimensions_t<energy_dimension, volume_dimension>;
using energy_density_rate_dimension = divide_dimensions_t<energy_density_dimension, time_dimension>;
using pressure_gradient_dimension = divide_dimensions_t<pressure_dimension, length_dimension>;

#ifndef HPC_DISABLE_DIMENSIONAL_ANALYSIS

template <class T, class Dimension>
class quantity {
  T m_impl;
  public:
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr quantity(T in) noexcept : m_impl(in) {}
  HPC_ALWAYS_INLINE constexpr quantity() noexcept = default;
  HPC_ALWAYS_INLINE constexpr quantity(quantity const& in) noexcept = default;
  HPC_ALWAYS_INLINE quantity& operator=(quantity const& in) noexcept = default;
  template <class Dimension2>
  HPC_ALWAYS_INLINE constexpr explicit quantity(quantity<T, Dimension2> const& in) noexcept : quantity(T(in)) {}
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr explicit operator T() const noexcept { return m_impl; }
  HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr T get() const noexcept { return m_impl; }
};

template <class T, class D>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr auto
operator+(quantity<T, D> left, quantity<T, D> right) noexcept
{
  return quantity<T, D>(T(left) + T(right));
}

template <class T, class D>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE quantity<T, D>&
operator+=(quantity<T, D>& left, quantity<T, D> right) noexcept
{
  left = left + right;
  return left;
}

template <class T, class D>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr auto
operator+(quantity<T, D> left, T right) noexcept
{
  return quantity<T, D>(T(left) + right);
}

template <class T, class D>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE quantity<T, D>&
operator+=(quantity<T, D>& left, T right) noexcept
{
  left = left + right;
  return left;
}

template <class T, class D>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr auto
operator+(T left, quantity<T, D> right) noexcept
{
  return right + left;
}

template <class T, class D>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr quantity<T, D>
operator-(quantity<T, D> x) noexcept
{
  return quantity<T, D>(-T(x));
}

template <class T, class D>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr auto
operator-(quantity<T, D> left, quantity<T, D> right) noexcept
{
  return quantity<T, D>(T(left) - T(right));
}

template <class T, class D>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE quantity<T, D>&
operator-=(quantity<T, D>& left, quantity<T, D> right) noexcept
{
  left = left - right;
  return left;
}

template <class T, class D>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr auto
operator-(quantity<T, D> left, T right) noexcept
{
  return quantity<T, D>(T(left) - right);
}

template <class T, class D>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE quantity<T, D>&
operator-=(quantity<T, D>& left, T right) noexcept
{
  left = left - right;
  return left;
}

template <class T, class D>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr auto
operator-(T left, quantity<T, D> right) noexcept
{
  return quantity<T, D>(left - T(right));
}

template <class T, class LD, class RD>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr auto
operator*(quantity<T, LD> left, quantity<T, RD> right) noexcept
{
  return quantity<T, multiply_dimensions_t<LD, RD>>(T(left) * T(right));
}

template <class T, class D>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE quantity<T, D>&
operator*=(quantity<T, D>& left, quantity<T, no_dimension> right) noexcept
{
  left = left * right;
  return left;
}

template <class T, class D>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr auto
operator*(quantity<T, D> left, T right) noexcept
{
  return quantity<T, D>(T(left) * right);
}

template <class T, class D>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE quantity<T, D>&
operator*=(quantity<T, D>& left, T right) noexcept
{
  left = left * right;
  return left;
}

template <class T, class D>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr auto
operator*(T left, quantity<T, D> right) noexcept
{
  return right * left;
}

template <class T, class LD, class RD>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr auto
operator/(quantity<T, LD> left, quantity<T, RD> right) noexcept
{
  return quantity<T, divide_dimensions_t<LD, RD>>(T(left) / T(right));
}

template <class T, class LD, class RD>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE quantity<T, LD>&
operator/=(quantity<T, LD>& left, quantity<T, RD> right) noexcept
{
  left = left / right;
  return left;
}

template <class T, class D>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr auto
operator/(quantity<T, D> left, T right) noexcept
{
  return quantity<T, D>(T(left) / right);
}

template <class T, class LD, class RD>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE quantity<T, LD>&
operator/=(quantity<T, LD>& left, T right) noexcept
{
  left = left / right;
  return left;
}

template <class T, class D>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr auto
operator/(T left, quantity<T, D> right) noexcept
{
  return quantity<T, divide_dimensions_t<no_dimension, D>>(left / T(right));
}

template <class T, class D>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr bool
operator==(quantity<T, D> left, quantity<T, D> right) noexcept {
  return T(left) == T(right);
}

template <class T, class D>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr bool
operator==(quantity<T, D> left, T right) noexcept {
  return T(left) == right;
}

template <class T, class D>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr bool
operator==(T left, quantity<T, D> right) noexcept {
  return left == T(right);
}

template <class T, class D>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr bool
operator!=(quantity<T, D> left, quantity<T, D> right) noexcept {
  return T(left) != T(right);
}

template <class T, class D>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr bool
operator!=(quantity<T, D> left, T right) noexcept {
  return T(left) != right;
}

template <class T, class D>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr bool
operator!=(T left, quantity<T, D> right) noexcept {
  return left != T(right);
}

template <class T, class D>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr bool
operator>(quantity<T, D> left, quantity<T, D> right) noexcept {
  return T(left) > T(right);
}

template <class T, class D>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr bool
operator>(quantity<T, D> left, T right) noexcept {
  return T(left) > right;
}

template <class T, class D>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr bool
operator>(T left, quantity<T, D> right) noexcept {
  return left > T(right);
}

template <class T, class D>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr bool
operator<(quantity<T, D> left, quantity<T, D> right) noexcept {
  return T(left) < T(right);
}

template <class T, class D>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr bool
operator<(quantity<T, D> left, T right) noexcept {
  return T(left) < right;
}

template <class T, class D>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr bool
operator<(T left, quantity<T, D> right) noexcept {
  return left < T(right);
}

template <class T, class D>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr bool
operator>=(quantity<T, D> left, quantity<T, D> right) noexcept {
  return T(left) >= T(right);
}

template <class T, class D>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr bool
operator>=(quantity<T, D> left, T right) noexcept {
  return T(left) >= right;
}

template <class T, class D>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr bool
operator>=(T left, quantity<T, D> right) noexcept {
  return left >= T(right);
}

template <class T, class D>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr bool
operator<=(quantity<T, D> left, quantity<T, D> right) noexcept {
  return T(left) <= T(right);
}

template <class T, class D>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr bool
operator<=(quantity<T, D> left, T right) noexcept {
  return T(left) <= right;
}

template <class T, class D>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE constexpr bool
operator<=(T left, quantity<T, D> right) noexcept {
  return left <= T(right);
}

template <class T, class D>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE auto sqrt(quantity<T, D> x) noexcept
{
  using std::sqrt;
  return quantity<T, root_dimension_t<D, 2>>(sqrt(T(x)));
}

template <class T, class D>
HPC_ALWAYS_INLINE HPC_HOST_DEVICE auto cbrt(quantity<T, D> x) noexcept
{
  using std::cbrt;
  return quantity<T, root_dimension_t<D, 3>>(cbrt(T(x)));
}

#else

template <class T, class D>
using quantity = T;

#endif

template <class T>
using dimensionless = quantity<T, no_dimension>;
template <class T>
using length = quantity<T, length_dimension>;
template <class T>
using mass = quantity<T, mass_dimension>;
template <class T>
using time = quantity<T, time_dimension>;
template <class T>
using temperature = quantity<T, temperature_dimension>;
template <class T>
using speed = quantity<T, speed_dimesion>;
template <class T>
using area = quantity<T, area_dimension>;
template <class T>
using volume = quantity<T, volume_dimension>;
template <class T>
using position = vector3<length<T>>;
template <class T>
using displacement = vector3<length<T>>;
template <class T>
using velocity = vector3<speed<T>>;
template <class T>
using acceleration = vector3<quantity<T, acceleration_dimension>>;
template <class T>
using basis_gradient = vector3<quantity<T, gradient_dimension>>;
template <class T>
using deformation_gradient = matrix3x3<quantity<T, no_dimension>>;
template <class T>
using pressure = quantity<T, pressure_dimension>;
template <class T>
using symmetric_stress = symmetric3x3<pressure<T>>;
template <class T>
using symmetric_deformation = symmetric3x3<quantity<T, no_dimension>>;
template <class T>
using force = vector3<quantity<T, force_dimension>>;
template <class T>
using energy = quantity<T, energy_dimension>;
template <class T>
using power = quantity<T, power_dimension>;
template <class T>
using specific_energy = quantity<T, specific_energy_dimension>;
template <class T>
using specific_energy_rate = quantity<T, specific_energy_rate_dimension>;
template <class T>
using density = quantity<T, density_dimension>;
template <class T>
using dynamic_viscosity = quantity<T, dynamic_viscosity_dimension>;
template <class T>
using kinematic_viscosity = quantity<T, kinematic_viscosity_dimension>;
template <class T>
using velocity_gradient = matrix3x3<quantity<T, velocity_gradient_dimension>>;
template <class T>
using symmetric_velocity_gradient = symmetric3x3<quantity<T, velocity_gradient_dimension>>;
template <class T>
using heat_flux = vector3<quantity<T, heat_flux_dimension>>;
template <class T>
using pressure_rate = quantity<T, pressure_rate_dimension>;
template <class T>
using energy_density_rate = quantity<T, energy_density_rate_dimension>;
template <class T>
using pressure_gradient = vector3<quantity<T, pressure_gradient_dimension>>;

}
