#ifndef LGR_NODE_SCALAR_HPP
#define LGR_NODE_SCALAR_HPP

#include <Omega_h_teuchos.hpp>

namespace lgr {

struct Simulation;
struct Scalar;

Scalar* node_scalar_factory(Simulation& sim, std::string const&, Teuchos::ParameterList& pl);

}

#endif
