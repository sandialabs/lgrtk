#pragma once

#include <lgr_mesh_indices.hpp>
#include <hpc_vector.hpp>

namespace lgr {

class input;
class state;

void update_sigma_with_p_h(state& s, material_index const material);
void update_p_h(state& s, double const dt,
    material_index const material,
    hpc::device_vector<double, node_index> const& old_p_h_vector);
void update_e_h(state& s, double const dt,
    material_index const material,
    hpc::device_vector<double, node_index> const& old_e_h_vector);
void nodal_ideal_gas(input const& in, state& s, material_index const);
void update_nodal_density(state& s, material_index const);
void interpolate_K(state& s, material_index const);
void interpolate_rho(state& s, material_index const);
void update_p_h_dot_from_a(input const& in, state& s, material_index const material);
void update_e_h_dot_from_a(input const& in, state& s, material_index const material);

}
