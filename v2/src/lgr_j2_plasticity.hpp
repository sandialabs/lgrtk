#ifndef LGR_J2_PLASTICITY_HPP
#define LGR_J2_PLASTICITY_HPP

#include <lgr_element_types.hpp>
#include <lgr_model.hpp>
#include <string>

namespace lgr {

struct Properties;

void read_and_validate_elastic_params(
    Omega_h::InputMap & pl, Properties & props);

void read_and_validate_hardening_params(
    Omega_h::InputMap & pl, Properties & props);

void read_and_validate_rate_sensitivity_params(
    Omega_h::InputMap & pl, Properties & props);

template<class Elem>
ModelBase* j2_plasticity_factory(
    Simulation& sim, std::string const& name, Omega_h::InputMap& pl);

#define LGR_EXPL_INST(Elem)                                                    \
  extern template ModelBase* j2_plasticity_factory<Elem>(                      \
      Simulation&, std::string const&, Omega_h::InputMap&);
LGR_EXPL_INST_ELEMS
#undef LGR_EXPL_INST

}
  // namespace lgr

#endif
