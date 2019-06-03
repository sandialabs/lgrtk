#pragma once

#include <lgr_mesh_indices.hpp>

namespace lgr {

class state;

void initialize_composite_tetrahedron_V(state& s);
void initialize_composite_tetrahedron_grad_N(state& s);
void update_composite_tetrahedron_h_min(state& s);
void update_nodal_mass_composite_tetrahedron(state& s, material_index const material);

}
