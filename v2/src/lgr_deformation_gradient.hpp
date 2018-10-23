#ifndef LGR_DEFORMATION_GRADIENT_HPP 
#define LGR_DEFORMATION_GRADIENT_HPP 

#include <lgr_element_types.hpp>
#include <lgr_model.hpp>
#include <string>

namespace lgr {

template <class Elem>
ModelBase* deformation_gradient_factory(
    Simulation& sim, std::string const&,
    Omega_h::InputMap&);

#define LGR_EXPL_INST(Elem) \
extern template ModelBase* \
deformation_gradient_factory<Elem>( \
    Simulation&, std::string const&, Omega_h::InputMap&);
LGR_EXPL_INST_ELEMS
#undef LGR_EXPL_INST

}

#endif

