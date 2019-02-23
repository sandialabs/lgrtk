#ifndef LGR_RUN_HPP
#define LGR_RUN_HPP

#include <Omega_h_input.hpp>
#include <lgr_factories.hpp>
#include <lgr_setup.hpp>

namespace lgr {

int run_cmdline(int argc, char** argv,
    std::function<void(lgr::Factories&, std::string const&)> add_factories,
    std::function<void(lgr::Setups&)> add_other_setups
    );

}  // namespace lgr

#endif
