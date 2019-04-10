#pragma once

#include <map>

#include <lgr_counting_range.hpp>
#include <lgr_physics_types.hpp>
#include <lgr_range_sum.hpp>
#include <lgr_vector3.hpp>
#include <lgr_matrix3x3.hpp>
#include <lgr_symmetric3x3.hpp>
#include <lgr_host_vector.hpp>

namespace lgr {

class state {
  public:
  int n = 0;
  double time = 0.0;
  counting_range<element_index> elements{element_index(0)};
  counting_range<node_in_element_index> nodes_in_element{node_in_element_index(0)};
  counting_range<node_index> nodes{node_index(0)};
  counting_range<point_in_element_index> points_in_element{point_in_element_index(1)};
  counting_range<point_index> points{point_index(0)};
  device_memory_pool devpool;
  device_vector<node_index, element_node_index> elements_to_nodes{devpool};
  range_sum<node_element_index, device_allocator<node_element_index>, node_index> nodes_to_node_elements{device_allocator<node_element_index>{devpool}};
  device_vector<element_index, node_element_index> node_elements_to_elements{devpool};
  device_vector<node_in_element_index, node_element_index> node_elements_to_nodes_in_element{devpool};
  device_vector<vector3<double>, node_index> x{devpool}; // current nodal positions
  device_vector<vector3<double>, node_index> u{devpool}; // nodal displacements since previous time state
  device_vector<vector3<double>, node_index> v{devpool}; // nodal velocities
  device_vector<vector3<double>, node_index> old_v{devpool}; // nodal velocities at previous time state
  device_vector<double, point_index> V{devpool}; // measures (volume/area/length)
  device_vector<vector3<double>, point_node_index> grad_N{devpool}; // gradients of basis functions
  device_vector<matrix3x3<double>, point_index> F_total{devpool}; // deformation gradient since simulation start
  device_vector<symmetric3x3<double>, point_index> sigma{devpool}; // Cauchy stress tensor
  device_vector<symmetric3x3<double>, point_index> symm_grad_v{devpool}; // symmetrized gradient of velocity
  device_vector<double, point_index> p{devpool}; // pressure at elements (output only!)
  device_vector<vector3<double>, point_index> v_prime{devpool}; // fine-scale velocity
  device_vector<vector3<double>, point_index> q{devpool}; // element-center heat flux
  device_vector<double, point_node_index> W{devpool}; // work done, per element-node pair (contribution to a node's work by an element)
  device_vector<double, node_index> p_h_dot{devpool}; // time derivative of stabilized nodal pressure
  device_vector<double, node_index> p_h{devpool}; // stabilized nodal pressure
  device_vector<double, node_index> old_p_h{devpool}; // stabilized nodal pressure at previous time state
  device_vector<double, point_index> K{devpool}; // (tangent/effective) bulk modulus
  device_vector<double, node_index> K_h{devpool}; // (tangent/effective) bulk modulus at nodes
  device_vector<double, point_index> G{devpool}; // (tangent/effective) shear modulus
  device_vector<double, point_index> c{devpool}; // sound speed / plane wave speed
  device_vector<vector3<double>, point_node_index> element_f{devpool}; // (internal) force per element-node pair (contribution to a node's force by an element)
  device_vector<vector3<double>, node_index> f{devpool}; // nodal (internal) forces
  device_vector<double, point_index> rho{devpool}; // element density
  device_vector<double, point_index> e{devpool}; // element specific internal energy
  device_vector<double, point_index> old_e{devpool}; // specific internal energy at previous time state
  device_vector<double, point_index> rho_e_dot{devpool}; // time derivative of internal energy density
  device_vector<double, node_index> m{devpool}; // nodal mass
  device_vector<vector3<double>, node_index> a{devpool}; // nodal acceleration
  device_vector<double, element_index> h_min{devpool}; // minimum characteristic element length, used for stable time step
  device_vector<double, element_index> h_art{devpool}; // characteristic element length used for artificial viscosity
  device_vector<double, point_index> nu_art{devpool}; // artificial kinematic viscosity scalar
  device_vector<double, point_index> element_dt{devpool}; // stable time step of each element
  device_vector<double, node_index> e_h{devpool}; // nodal specific internal energy
  device_vector<double, node_index> old_e_h{devpool}; // nodal specific internal energy at previous time state
  device_vector<double, node_index> e_h_dot{devpool}; // time derivative of nodal specific internal energy
  device_vector<double, node_index> rho_h{devpool}; // nodal density
  device_vector<material_index, element_index> material{devpool}; // element material
  device_vector<double, element_index> Q{devpool}; // element quality
  std::map<std::string, device_vector<node_index, int>> node_sets;
  host_vector<device_vector<element_index, int>, material_index> element_sets;
  double next_file_output_time;
  double dt = 0.0;
  double max_stable_dt;
};

}
