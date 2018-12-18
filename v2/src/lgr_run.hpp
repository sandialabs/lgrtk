#ifndef LGR_RUN_HPP
#define LGR_RUN_HPP

#include <Omega_h_input.hpp>
#include <lgr_factories.hpp>

namespace lgr {

void run(Omega_h::CommPtr comm, Omega_h::InputMap& pl,
    Factories&& model_factories = Factories());

}

#endif
