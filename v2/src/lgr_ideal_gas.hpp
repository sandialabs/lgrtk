#ifndef LGR_IDEAL_GAS_HPP
#define LGR_IDEAL_GAS_HPP

#include <lgr_element_types.hpp>
#include <lgr_model.hpp>
#include <string>

namespace lgr {

OMEGA_H_INLINE void ideal_gas_update(
    double gamma,
    double density,
    double specific_internal_energy,
    double& pressure,
    double& wave_speed) {
  OMEGA_H_CHECK(density > 0.0);
  OMEGA_H_CHECK(specific_internal_energy > 0.0);
  pressure =
    (gamma - 1.) * density * specific_internal_energy;
  auto I = identity_matrix<3, 3>();
  auto bulk_modulus = gamma * pressure;
  wave_speed = std::sqrt(bulk_modulus / density);
  OMEGA_H_CHECK(wave_speed > 0.0);
}

template <class Elem>
ModelBase* ideal_gas_factory(Simulation& sim, std::string const& name, Omega_h::InputMap& pl);

#define LGR_EXPL_INST(Elem) \
extern template ModelBase* ideal_gas_factory<Elem>(Simulation&, std::string const&, Omega_h::InputMap&);
LGR_EXPL_INST_ELEMS
#undef LGR_EXPL_INST

}

#endif
