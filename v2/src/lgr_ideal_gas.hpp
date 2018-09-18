#ifndef LGR_PERFECT_GAS_HPP
#define LGR_PERFECT_GAS_HPP

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
  pressure =
    (gamma - 1.) * density * specific_internal_energy;
  auto I = identity_matrix<3, 3>();
  auto bulk_modulus = gamma * pressure;
  wave_speed = std::sqrt(bulk_modulus / density);
}

template <class Elem>
ModelBase* ideal_gas_factory(Simulation& sim, std::string const& name, Teuchos::ParameterList& pl);

#define LGR_EXPL_INST(Elem) \
extern template ModelBase* ideal_gas_factory<Elem>(Simulation&, std::string const&, Teuchos::ParameterList&);
LGR_EXPL_INST_ELEMS
#undef LGR_EXPL_INST

}

#endif
