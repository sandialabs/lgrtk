#ifndef LGR_LINEAR_ELASTIC_HPP
#define LGR_LINEAR_ELASTIC_HPP

#include <lgr_element_types.hpp>
#include <lgr_model.hpp>
#include <string>

namespace lgr {

template <class Elem>
ModelBase* linear_elastic_factory(Simulation& sim, std::string const& name, Teuchos::ParameterList& pl);

#define LGR_EXPL_INST(Elem) \
extern template ModelBase* linear_elastic_factory<Elem>(Simulation&, std::string const&, Teuchos::ParameterList&);
LGR_EXPL_INST_ELEMS
#undef LGR_EXPL_INST

}

#endif
