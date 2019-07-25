#ifndef LGR_HYPER_EP_DEFS_HPP
#define LGR_HYPER_EP_DEFS_HPP

#include "common.hpp"

namespace lgr {
namespace hyper_ep {

constexpr Hardening props_hardening = Hardening::JOHNSON_COOK;
constexpr RateDependence props_rate_dep = RateDependence::NONE;
constexpr Damage props_damage = Damage::NONE;
constexpr Elastic props_elastic = Elastic::NEO_HOOKEAN;
constexpr bool props_allow_no_tension = false;
constexpr bool props_allow_no_shear = false;
constexpr bool props_set_stress_to_zero = false;

} // namepsace hyper_ep
} // namepsace lgr

#endif // LGR_HYPER_EP_DEFS_HPP
