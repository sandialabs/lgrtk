#ifndef LGR_CMDLINE_HIST_HPP
#define LGR_CMDLINE_HIST_HPP

#include <Omega_h_input.hpp>

namespace lgr {

struct Simulation;

void setup_cmdline_hist(Simulation&, Omega_h::InputMap&);

}  // namespace lgr

#endif
