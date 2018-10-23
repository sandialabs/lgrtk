#ifndef LGR_VTK_OUTPUT_HPP
#define LGR_VTK_OUTPUT_HPP

#include <Omega_h_input.hpp>

namespace lgr {

struct Response;
struct Simulation;

Response* vtk_output_factory(Simulation& sim, std::string const&, Omega_h::InputMap& pl);

}

#endif
