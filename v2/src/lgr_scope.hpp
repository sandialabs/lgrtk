#ifndef LGR_SCOPE_HPP
#define LGR_SCOPE_HPP

#include <Omega_h_stack.hpp>

namespace lgr {

struct Simulation;

struct Scope {
  Simulation& sim;
  char const* name;
  Omega_h::ScopedTimer timer;
  Scope(Simulation& sim_in, char const* name_in)
    :sim(sim_in)
    ,name(name_in)
    ,timer(name)
  {}
  ~Scope();
};

}

#define LGR_SCOPE(sim) ::lgr::Scope scope(sim, __FUNCTION__)

#endif
