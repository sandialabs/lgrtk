#ifndef LGR_VTK_OUTPUT_HPP
#define LGR_VTK_OUTPUT_HPP

#include <Omega_h_teuchos.hpp>

namespace lgr {

struct Response;
struct Simulation;

Response* vtk_output_factory(Simulation& sim, std::string const&, Teuchos::ParameterList& pl);

}

#endif
