#pragma once

#include <lgr_mesh_indices.hpp>
#include <hpc_vector.hpp>
#include <hpc_dimensional.hpp>

namespace lgr {

class input;
class state;

void update_sigma_with_p_h(state& s, material_index const material);
void update_sigma_with_p_h_p_prime(input const& in, state& s, material_index const material,
    hpc::time<double> const dt,
    hpc::device_vector<hpc::pressure<double>, node_index> const& old_p_h_vector);
void update_p_h(state& s, hpc::time<double> const dt,
    material_index const material,
    hpc::device_vector<hpc::pressure<double>, node_index> const& old_p_h_vector);
void update_e_h(state& s, hpc::time<double> const dt,
    material_index const material,
    hpc::device_vector<hpc::specific_energy<double>, node_index> const& old_e_h_vector);
void nodal_ideal_gas(input const& in, state& s, material_index const);
void update_nodal_density(state& s, material_index const);
void interpolate_K(state& s, material_index const);
void interpolate_rho(state& s, material_index const);
void update_p_h_dot_from_a(input const& in, state& s, material_index const material);
void update_e_h_dot_from_a(input const& in, state& s, material_index const material);

}
