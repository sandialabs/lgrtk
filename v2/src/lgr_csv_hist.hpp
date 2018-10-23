#ifndef LGR_CSV_HIST_HPP
#define LGR_CSV_HIST_HPP

#include <Omega_h_input.hpp>

namespace lgr {

struct Response;
struct Simulation;

Response* csv_hist_factory(Simulation& sim, std::string const&,
    Omega_h::InputMap& pl);

}

#endif

