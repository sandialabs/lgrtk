#ifndef LGR_ARTIFICIAL_VISCOSITY_HPP
#define LGR_ARTIFICIAL_VISCOSITY_HPP

#include <lgr_element_types.hpp>
#include <lgr_model.hpp>
#include <string>

namespace lgr {

template <class Elem>
ModelBase* artificial_viscosity_factory(
    Simulation& sim, std::string const& name, Omega_h::InputMap& pl);

#define LGR_EXPL_INST(Elem)                                                    \
  extern template ModelBase* artificial_viscosity_factory<Elem>(               \
      Simulation&, std::string const&, Omega_h::InputMap&);
LGR_EXPL_INST_ELEMS
#undef LGR_EXPL_INST

}  // namespace lgr

#endif
