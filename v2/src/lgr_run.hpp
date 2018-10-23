#ifndef LGR_RUN_HPP
#define LGR_RUN_HPP

#include <lgr_factories.hpp>
#include <Omega_h_input.hpp>

namespace lgr {

void run(Omega_h::CommPtr comm, Omega_h::InputMap& pl,
    Factories&& model_factories = Factories());

}

#endif
