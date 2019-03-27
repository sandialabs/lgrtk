#pragma once

#include <map>

#include <lgr_int_range.hpp>
#include <lgr_physics_types.hpp>
#include <lgr_int_range_sum.hpp>
#include <lgr_vector3.hpp>
#include <lgr_matrix3x3.hpp>
#include <lgr_symmetric3x3.hpp>

namespace lgr {

class state {
  public:
  int n = 0;
  double time = 0.0;
  counting_range<element_index> elements{0};
  counting_range<node_in_element_index> nodes_in_element{0};
  counting_range<node_index> nodes{0};
  host_vector<node_index, element_node_index> elements_to_nodes;
  int_range_sum<host_allocator<int>> nodes_to_node_elements{host_allocator<int>{}};
  device_vector<element_index, int> node_elements_to_elements;
  device_vector<node_in_element_index, int> node_elements_to_nodes_in_element;
  device_vector<vector3<double>, node_index> x; // current nodal positions
  device_vector<vector3<double>, node_index> u; // nodal displacements since previous time state
  device_vector<vector3<double>, node_index> v; // nodal velocities
  device_vector<vector3<double>, node_index> old_v; // nodal velocities at previous time state
  device_vector<double, element_index> V; // element measures (volume/area/length)
  device_vector<vector3<double>, element_node_index> grad_N; // gradients of basis functions
  device_vector<matrix3x3<double>, element_index> F_total; // deformation gradient since simulation start
  device_vector<symmetric3x3<double>, element_index> sigma; // Cauchy stress tensor
  device_vector<symmetric3x3<double>, element_index> symm_grad_v; // symmetrized gradient of velocity
  device_vector<double, element_index> p; // pressure at elements (output only!)
  device_vector<vector3<double>, element_index> v_prime; // fine-scale velocity
  device_vector<vector3<double>, node_index> q; // nodal heat flux
  device_vector<double, element_node_index> W; // work done, per element-node pair (contribution to a node's work by an element)
  device_vector<double, node_index> p_h_dot; // time derivative of stabilized nodal pressure
  device_vector<double, node_index> p_h; // stabilized nodal pressure
  device_vector<double, node_index> old_p_h; // stabilized nodal pressure at previous time state
  device_vector<double, element_index> K; // (tangent/effective) bulk modulus
  device_vector<double, element_index> G; // (tangent/effective) shear modulus
  device_vector<double, element_index> c; // sound speed / plane wave speed
  device_vector<vector3<double>, element_node_index> element_f; // (internal) force per element-node pair (contribution to a node's force by an element)
  device_vector<vector3<double>, node_index> f; // nodal (internal) forces
  device_vector<double, element_index> rho; // element density
  device_vector<double, element_index> e; // element specific internal energy
  device_vector<double, element_index> old_e; // specific internal energy at previous time state
  device_vector<double, element_index> rho_e_dot; // time derivative of internal energy density
  device_vector<double, node_index> m; // nodal mass
  device_vector<vector3<double>, node_index> a; // nodal acceleration
  device_vector<double, element_index> h_min; // minimum characteristic element length, used for stable time step
  device_vector<double, element_index> h_art; // characteristic element length used for artificial viscosity
  device_vector<double, element_index> nu_art; // artificial kinematic viscosity scalar
  device_vector<double, element_index> element_dt; // stable time step of each element
  device_vector<double, node_index> e_h; // nodal specific internal energy
  device_vector<double, node_index> old_e_h; // nodal specific internal energy at previous time state
  device_vector<double, node_index> e_h_dot; // time derivative of nodal specific internal energy
  device_vector<double, node_index> rho_h; // nodal density
  std::map<std::string, device_vector<node_index, int>> node_sets;
  double next_file_output_time;
  double dt = 0.0;
  double max_stable_dt;
};

}
