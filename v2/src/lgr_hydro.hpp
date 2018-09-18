#ifndef LGR_HYDRO_HPP
#define LGR_HYDRO_HPP

#include <lgr_element_types.hpp>

namespace lgr {

struct Simulation;

template <class Elem>
void initialize_configuration(Simulation& sim);
template <class Elem>
void lump_masses(Simulation& sim);
template <class Elem>
void update_position(Simulation& sim);
template <class Elem>
void update_configuration(Simulation& sim);
template <class Elem>
void correct_velocity(Simulation& sim);
template <class Elem>
void compute_stress_divergence(Simulation& sim);
template <class Elem>
void compute_nodal_acceleration(Simulation& sim);
template <class Elem>
void compute_point_time_steps(Simulation& sim);

void apply_tractions(Simulation& sim);

#define LGR_EXPL_INST(Elem) \
extern template void initialize_configuration<Elem>(Simulation& sim); \
extern template void lump_masses<Elem>(Simulation& sim); \
extern template void update_position<Elem>(Simulation& sim); \
extern template void update_configuration<Elem>(Simulation& sim); \
extern template void correct_velocity<Elem>(Simulation& sim); \
extern template void compute_stress_divergence<Elem>(Simulation& sim); \
extern template void compute_nodal_acceleration<Elem>(Simulation& sim); \
extern template void compute_point_time_steps<Elem>(Simulation& sim);
LGR_EXPL_INST_ELEMS
#undef LGR_EXPL_INST

}

#endif
