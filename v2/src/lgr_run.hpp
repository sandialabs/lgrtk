#ifndef LGR_RUN_HPP
#define LGR_RUN_HPP

#include <Omega_h_input.hpp>
#include <lgr_factories.hpp>

namespace lgr {

void run(
    Omega_h::CommPtr comm, Omega_h::InputMap& pl, Factories&& model_factories);
int run_cmdline(int argc, char** argv,
    std::function<void(lgr::Factories&, std::string const&)> add_factories);

}  // namespace lgr

#endif
