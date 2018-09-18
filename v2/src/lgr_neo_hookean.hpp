#ifndef LGR_NEO_HOOKEAN_HPP
#define LGR_NEO_HOOKEAN_HPP

#include <lgr_element_types.hpp>
#include <lgr_model.hpp>
#include <string>

namespace lgr {

template <class Elem>
ModelBase* neo_hookean_factory(Simulation& sim, std::string const& name, Teuchos::ParameterList& pl);

#define LGR_EXPL_INST(Elem) \
extern template ModelBase* neo_hookean_factory<Elem>(Simulation&, std::string const&, Teuchos::ParameterList&);
LGR_EXPL_INST_ELEMS
#undef LGR_EXPL_INST

}

#endif
