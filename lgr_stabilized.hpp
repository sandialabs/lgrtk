#pragma once

#include <lgr_mesh_indices.hpp>

namespace lgr {

class input;
class state;

void update_sigma_with_p_h(state& s, material_index const material);
void update_v_prime(input const& in, state& s, material_index const material);
void update_q(input const& in, state& s, material_index const material);
void update_p_h_W(state& s);
void update_e_h_W(state& s);
void update_p_h_dot(state& s, material_index const);
void update_e_h_dot(state& s, material_index const);
void nodal_ideal_gas(input const& in, state& s, material_index const);
void update_nodal_density(state& s, material_index const);
void interpolate_K(state& s, material_index const);
void interpolate_rho(state& s, material_index const);

}
