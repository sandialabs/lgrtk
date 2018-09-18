#ifndef LGR_CSV_HIST_HPP
#define LGR_CSV_HIST_HPP

#include <Omega_h_teuchos.hpp>

namespace lgr {

struct Response;
struct Simulation;

Response* csv_hist_factory(Simulation& sim, std::string const&,
    Teuchos::ParameterList& pl);

}

#endif

