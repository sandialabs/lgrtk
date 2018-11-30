#ifndef LGR_STABILIZATION_HPP 
#define LGR_STABILIZATION_HPP 

#include <lgr_element_types.hpp>
#include <lgr_model.hpp>
#include <string>

namespace lgr {

template <class Elem>
ModelBase* stabilization_factory(
    Simulation& sim, std::string const&,
    Omega_h::InputMap&);

#define LGR_EXPL_INST(Elem) \
extern template ModelBase* \
stabilization_factory<Elem>( \
    Simulation&, std::string const&, Omega_h::InputMap&);
LGR_EXPL_INST_ELEMS
#undef LGR_EXPL_INST

}

#endif

