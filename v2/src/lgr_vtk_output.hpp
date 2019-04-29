#ifndef LGR_VTK_OUTPUT_HPP
#define LGR_VTK_OUTPUT_HPP

#include <Omega_h_input.hpp>

namespace lgr {

struct Response;
struct Simulation;

using LgrFields = std::set<std::size_t>;
using OshFields = std::set<std::string>;

Response* vtk_output_factory(
    Simulation& sim, std::string const&, Omega_h::InputMap& pl);

void write_vtu(Omega_h::filesystem::path const& step_path,
    Simulation& sim, bool compress, LgrFields lgr_fields[4],
    OshFields osh_fields[4], bool override_path = false);

}  // namespace lgr

#endif
