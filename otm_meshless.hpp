#pragma once

#include <lgr_mesh_indices.hpp>

namespace lgr {

class state;
class input;

void otm_initialize(input& in, state& s, std::string const& filename);
void otm_initialize_state(input const& in, state& s);
void otm_initialize_u(state& s);
void otm_initialize_F(state& s);
void otm_initialize_grad_val_N(state& s);
void otm_update_nodal_internal_force(state& s);
void otm_update_nodal_external_force(state& s);
void otm_update_nodal_force(state& s);
void otm_update_nodal_position(state& s);
void otm_update_point_position(state& s);
void otm_update_point_volume(state& s);
void otm_lump_nodal_mass(state& s);
void otm_update_reference(state& s);
void otm_nodal_linear_momentum(state& s);
void otm_update_material_state(input const& in, state& s, material_index const material);
void otm_run(std::string const& filename);
}
