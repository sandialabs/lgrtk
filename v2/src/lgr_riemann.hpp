#ifndef LGR_RIEMANN_HPP
#define LGR_RIEMANN_HPP

#include <Omega_h_array.hpp>

namespace lgr {

struct ExactRiemann {
  Omega_h::Reals velocity;
  Omega_h::Reals density;
  Omega_h::Reals pressure;
};

ExactRiemann exact_riemann(
    double left_density,
    double right_density,
    double left_pressure,
    double right_pressure,
    double shock_x,
    double gamma,
    double t,
    Omega_h::Reals x);

}

#endif
