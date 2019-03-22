#pragma once

#include <map>

#include <lgr_int_range.hpp>
#include <lgr_host_vector.hpp>
#include <lgr_int_range_sum.hpp>
#include <lgr_vector3.hpp>
#include <lgr_matrix3x3.hpp>
#include <lgr_symmetric3x3.hpp>

namespace lgr {

class state {
  public:
  int n = 0;
  double time = 0.0;
  int_range elements{0};
  int_range nodes_in_element{0};
  int_range nodes{0};
  host_vector<int> elements_to_nodes;
  int_range_sum<host_allocator<int>> nodes_to_node_elements{host_allocator<int>{}};
  host_vector<int> node_elements_to_elements;
  host_vector<int> node_elements_to_nodes_in_element;
  host_vector<vector3<double>> x; // current nodal positions
  host_vector<vector3<double>> u; // nodal displacements since previous time state
  host_vector<vector3<double>> v; // nodal velocities
  host_vector<vector3<double>> old_v; // nodal velocities at previous time state
  host_vector<double> V; // element measures (volume/area/length)
  host_vector<vector3<double>> grad_N; // gradients of basis functions
  host_vector<matrix3x3<double>> F_total; // deformation gradient since simulation start
  host_vector<symmetric3x3<double>> sigma; // Cauchy stress tensor
  host_vector<symmetric3x3<double>> symm_grad_v; // symmetrized gradient of velocity
  host_vector<double> p; // pressure at elements (output only!)
  host_vector<vector3<double>> v_prime; // fine-scale velocity
  host_vector<double> W; // work done, per element-node pair (contribution to a node's work by an element)
  host_vector<double> p_h_dot; // time derivative of stabilized nodal pressure
  host_vector<double> p_h; // stabilized nodal pressure
  host_vector<double> old_p_h; // stabilized nodal pressure at previous time state
  host_vector<double> K; // (tangent/effective) bulk modulus
  host_vector<double> G; // (tangent/effective) shear modulus
  host_vector<double> c; // sound speed / plane wave speed
  host_vector<vector3<double>> element_f; // (internal) force per element-node pair (contribution to a node's force by an element)
  host_vector<vector3<double>> f; // nodal (internal) forces
  host_vector<double> rho; // element density
  host_vector<double> e; // element specific internal energy
  host_vector<double> old_e; // specific internal energy at previous time state
  host_vector<double> m; // nodal mass
  host_vector<vector3<double>> a; // nodal acceleration
  host_vector<double> h_min; // minimum characteristic element length, used for stable time step
  host_vector<double> h_art; // characteristic element length used for artificial viscosity
  host_vector<double> nu_art; // artificial kinematic viscosity scalar
  host_vector<double> element_dt; // stable time step of each element
  std::map<std::string, host_vector<int>> node_sets;
  double next_file_output_time;
  double dt = 0.0;
  double max_stable_dt;
};

}
