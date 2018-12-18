#ifndef LGR_TRACTION_HPP
#define LGR_TRACTION_HPP

namespace lgr {

struct Simulation;

bool has_traction(Simulation& sim);
void apply_tractions(Simulation& sim);

}  // namespace lgr

#endif
