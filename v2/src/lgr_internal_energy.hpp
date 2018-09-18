#ifndef LGR_INTERNAL_ENERGY_HPP 
#define LGR_INTERNAL_ENERGY_HPP 

#include <lgr_element_types.hpp>
#include <lgr_model.hpp>
#include <string>

namespace lgr {

template <class Elem>
ModelBase* internal_energy_factory(
    Simulation& sim, std::string const&,
    Teuchos::ParameterList&);

#define LGR_EXPL_INST(Elem) \
extern template ModelBase* \
internal_energy_factory<Elem>( \
    Simulation&, std::string const&, Teuchos::ParameterList&);
LGR_EXPL_INST_ELEMS
#undef LGR_EXPL_INST

}

#endif
