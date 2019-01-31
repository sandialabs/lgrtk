#ifndef LGR_ANTI_LOCK_HPP
#define LGR_ANTI_LOCK_HPP

#include <lgr_element_types.hpp>
#include <lgr_model.hpp>
#include <string>

namespace lgr {

template <class Elem>
ModelBase* average_J_over_points_factory(
    Simulation& sim, std::string const&, Omega_h::InputMap&);
template <class Elem>
ModelBase* average_pressure_over_points_factory(
    Simulation& sim, std::string const&, Omega_h::InputMap&);
template <class Elem>
ModelBase* average_internal_energy_over_points_factory(
    Simulation& sim, std::string const&, Omega_h::InputMap&);
template <class Elem>
ModelBase* average_density_over_points_factory(
    Simulation& sim, std::string const&, Omega_h::InputMap&);
template <class Elem>
ModelBase* average_J_over_independent_set_factory(
    Simulation& sim, std::string const&, Omega_h::InputMap&);

#define LGR_EXPL_INST(Elem)                                                    \
  extern template ModelBase* average_J_over_points_factory<Elem>(                      \
      Simulation&, std::string const&, Omega_h::InputMap&); \
  extern template ModelBase* average_pressure_over_points_factory<Elem>(                      \
      Simulation&, std::string const&, Omega_h::InputMap&); \
  extern template ModelBase* average_internal_energy_over_points_factory<Elem>(                      \
      Simulation&, std::string const&, Omega_h::InputMap&); \
  extern template ModelBase* average_density_over_points_factory<Elem>(                      \
      Simulation&, std::string const&, Omega_h::InputMap&); \
  extern template ModelBase* average_J_over_independent_set_factory<Elem>(                      \
      Simulation&, std::string const&, Omega_h::InputMap&);
LGR_EXPL_INST_ELEMS
#undef LGR_EXPL_INST

}  // namespace lgr

#endif

