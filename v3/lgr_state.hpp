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
  counting_range<element_index> elements{element_index(0)};
  counting_range<node_in_element_index> nodes_in_element{node_in_element_index(0)};
  counting_range<node_index> nodes{node_index(0)};
  device_memory_pool mempool;
  device_vector<node_index, element_node_index> elements_to_nodes{mempool};
  int_range_sum<host_allocator<int>> nodes_to_node_elements{host_allocator<int>{}};
  device_vector<element_index, int> node_elements_to_elements{mempool};
  device_vector<node_in_element_index, int> node_elements_to_nodes_in_element{mempool};
  device_vector<vector3<double>, node_index> x{mempool}; // current nodal positions
  device_vector<vector3<double>, node_index> u{mempool}; // nodal displacements since previous time state
  device_vector<vector3<double>, node_index> v{mempool}; // nodal velocities
  device_vector<vector3<double>, node_index> old_v{mempool}; // nodal velocities at previous time state
  device_vector<double, element_index> V{mempool}; // element measures (volume/area/length)
  device_vector<vector3<double>, element_node_index> grad_N{mempool}; // gradients of basis functions
  device_vector<matrix3x3<double>, element_index> F_total{mempool}; // deformation gradient since simulation start
  device_vector<symmetric3x3<double>, element_index> sigma{mempool}; // Cauchy stress tensor
  device_vector<symmetric3x3<double>, element_index> symm_grad_v{mempool}; // symmetrized gradient of velocity
  device_vector<double, element_index> p{mempool}; // pressure at elements (output only!)
  device_vector<vector3<double>, element_index> v_prime{mempool}; // fine-scale velocity
  device_vector<vector3<double>, element_index> q{mempool}; // element-center heat flux
  device_vector<double, element_node_index> W{mempool}; // work done, per element-node pair (contribution to a node's work by an element)
  device_vector<double, node_index> p_h_dot{mempool}; // time derivative of stabilized nodal pressure
  device_vector<double, node_index> p_h{mempool}; // stabilized nodal pressure
  device_vector<double, node_index> old_p_h{mempool}; // stabilized nodal pressure at previous time state
  device_vector<double, element_index> K{mempool}; // (tangent/effective) bulk modulus
  device_vector<double, element_index> G{mempool}; // (tangent/effective) shear modulus
  device_vector<double, element_index> c{mempool}; // sound speed / plane wave speed
  device_vector<vector3<double>, element_node_index> element_f{mempool}; // (internal) force per element-node pair (contribution to a node's force by an element)
  device_vector<vector3<double>, node_index> f{mempool}; // nodal (internal) forces
  device_vector<double, element_index> rho{mempool}; // element density
  device_vector<double, element_index> e{mempool}; // element specific internal energy
  device_vector<double, element_index> old_e{mempool}; // specific internal energy at previous time state
  device_vector<double, element_index> rho_e_dot{mempool}; // time derivative of internal energy density
  device_vector<double, node_index> m{mempool}; // nodal mass
  device_vector<vector3<double>, node_index> a{mempool}; // nodal acceleration
  device_vector<double, element_index> h_min{mempool}; // minimum characteristic element length, used for stable time step
  device_vector<double, element_index> h_art{mempool}; // characteristic element length used for artificial viscosity
  device_vector<double, element_index> nu_art{mempool}; // artificial kinematic viscosity scalar
  device_vector<double, element_index> element_dt{mempool}; // stable time step of each element
  device_vector<double, node_index> e_h{mempool}; // nodal specific internal energy
  device_vector<double, node_index> old_e_h{mempool}; // nodal specific internal energy at previous time state
  device_vector<double, node_index> e_h_dot{mempool}; // time derivative of nodal specific internal energy
  device_vector<double, node_index> rho_h{mempool}; // nodal density
  std::map<std::string, device_vector<node_index, int>> node_sets;
  double next_file_output_time;
  double dt = 0.0;
  double max_stable_dt;
};

}
