#pragma once

namespace lgr {

enum base_quantity_t {
  LENGTH,
  MASS,
  TIME,
  CURRENT,
  TEMPERATURE,
  AMOUNT,
  INTENSITY
};

static constexpr int NUM_BASE_QUANTITIES = 7;

template
  <int length_power_in
  ,int mass_power_in
  ,int time_power_in
  ,int current_power_in
  ,int temperature_power_in
  ,int amount_power_in
  ,int intensity_power_in
  >
class dim_t {
  public:
  static constexpr int length_power = length_power_in;
  static constexpr int mass_power = mass_power_in;
  static constexpr int time_power = time_power_in;
  static constexpr int current_power = current_power_in;
  static constexpr int temperature_power = temperature_power_in;
  static constexpr int amount_power = amount_power_in;
  static constexpr int intensity_power = intensity_power_in;
};

class dimension_t {
  public:
  int powers[NUM_BASE_QUANTITIES];
};

template <class dim>
dimension_t get_runtime_dimension() {
  dimension_t out;
  out.powers[LENGTH] = dim::length_power;
  out.powers[MASS] = dim::mass_power;
  out.powers[TIME] = dim::time_power;
  out.powers[CURRENT] = dim::current_power;
  out.powers[TEMPERATURE] = dim::temperature_power;
  out.powers[AMOUNT] = dim::amount_power;
  out.powers[INTENSITY] = dim::intensity_power;
  return out;
}

template <class left, class right>
class multiply_dimensions_helper_t {
  public:
    using type = dim_t<
      left::length_power + right::length_power,
      left::mass_power + right::mass_power,
      left::time_power + right::time_power,
      left::current_power + right::current_power,
      left::temperature_power + right::temperature_power,
      left::amount_power + right::amount_power,
      left::intensity_power + right::intensity_power>;
};

template <class left, class right>
using multiply_dimensions_t = typename multiply_dimensions_helper_t<left, right>::type;

template <class left, class right>
class divide_dimensions_helper_t {
  public:
    using type = dim_t<
      left::length_power - right::length_power,
      left::mass_power - right::mass_power,
      left::time_power - right::time_power,
      left::current_power - right::current_power,
      left::temperature_power - right::temperature_power,
      left::amount_power - right::amount_power,
      left::intensity_power - right::intensity_power>;
};

template <class left, class right>
using divide_dimensions_t = typename divide_dimensions_helper_t<left, right>::type;

template <class dim, int factor>
class root_dimension_helper_t {
  public:
    using type = dim_t<
      dim::length_power / factor,
      dim::mass_power / factor,
      dim::time_power / factor,
      dim::current_power / factor,
      dim::temperature_power / factor,
      dim::amount_power / factor,
      dim::intensity_power / factor>;
};

template <class dim, int factor>
using root_dimension_t = typename root_dimension_helper_t<dim, factor>::type;

template <class dim, int factor>
class raise_dimension_helper_t {
  public:
    using type = dim_t<
      dim::length_power * factor,
      dim::mass_power * factor,
      dim::time_power * factor,
      dim::current_power * factor,
      dim::temperature_power * factor,
      dim::amount_power * factor,
      dim::intensity_power * factor>;
};

template <class dim, int factor>
using raise_dimension_t = typename raise_dimension_helper_t<dim, factor>::type;

}
