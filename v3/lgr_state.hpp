#pragma once

#include <map>

#include <hpc_range.hpp>
#include <hpc_range_sum.hpp>
#include <hpc_symmetric3x3.hpp>
#include <lgr_mesh_indices.hpp>
#include <lgr_material_set.hpp>
#include <hpc_array_vector.hpp>
#include <hpc_dimensional.hpp>

namespace lgr {

using dp_de_t = decltype(hpc::pressure<double>() / hpc::specific_energy<double>());
static_assert(std::is_same<dp_de_t, hpc::density<double>>::value, "dp_de should be a density");

class state {
  public:
  int n = 0;
  hpc::time<double> time = 0.0;
  hpc::counting_range<element_index> elements{element_index(0)};
  hpc::counting_range<node_in_element_index> nodes_in_element{node_in_element_index(0)};
  hpc::device_range_sum<point_node_index, point_index> nodes_in_support; // OTM: Support for each material point
  hpc::device_range_sum<point_in_influence_index, node_index> points_in_influence;
  hpc::counting_range<node_index> nodes{node_index(0)};
  hpc::counting_range<point_in_element_index> points_in_element{point_in_element_index(1)};
  hpc::counting_range<point_index> points{point_index(0)};
  hpc::device_vector<node_index, element_node_index> elements_to_nodes;
  hpc::device_vector<node_index, point_node_index> points_to_supported_nodes;
  hpc::device_vector<point_index, node_index> nodes_to_influenced_points;
  hpc::device_vector<node_in_support_index, node_point_index> node_influenced_points_to_supporting_nodes;
  hpc::device_range_sum<node_element_index, node_index> nodes_to_node_elements;
  hpc::device_vector<element_index, node_element_index> node_elements_to_elements;
  hpc::device_vector<node_in_element_index, node_element_index> node_elements_to_nodes_in_element;
  hpc::device_array_vector<hpc::position<double>, node_index> x; // current nodal positions
  hpc::device_array_vector<hpc::displacement<double>, node_index> u; // nodal displacements since previous time state
  hpc::device_array_vector<hpc::velocity<double>, node_index> v; // nodal velocities
  hpc::device_array_vector<hpc::momentum<double>, node_index> lm; // nodal linear momenta
  hpc::device_array_vector<hpc::position<double>, point_index> xm; // current material point positions
  hpc::device_vector<hpc::length<double>, point_index> h_otm; // characteristic length, used for max-ent functions
  hpc::device_vector<hpc::volume<double>, point_index> V; // integration point volumes
  hpc::device_vector<hpc::basis_value<double>, point_node_index> N; // values of basis functions
  hpc::device_array_vector<hpc::basis_gradient<double>, point_node_index> grad_N; // gradients of basis functions
  hpc::device_array_vector<hpc::deformation_gradient<double>, point_index> F_total; // deformation gradient since simulation start
  hpc::device_array_vector<hpc::symmetric_stress<double>, point_index> sigma; // Cauchy stress tensor
  hpc::device_array_vector<hpc::symmetric_velocity_gradient<double>, point_index> symm_grad_v; // symmetrized gradient of velocity
  hpc::device_vector<hpc::pressure<double>, point_index> p; // pressure at elements (output only!)
  hpc::device_array_vector<hpc::velocity<double>, point_index> v_prime; // fine-scale velocity
  hpc::device_vector<hpc::pressure<double>, point_index> p_prime; // fine-scale pressure
  hpc::device_array_vector<hpc::heat_flux<double>, point_index> q; // element-center heat flux
  hpc::device_vector<hpc::power<double>, point_node_index> W; // work done, per element-node pair (contribution to a node's work by an element)
  hpc::host_vector<hpc::device_vector<hpc::pressure_rate<double>, node_index>, material_index> p_h_dot; // time derivative of stabilized nodal pressure
  hpc::host_vector<hpc::device_vector<hpc::pressure<double>, node_index>, material_index> p_h; // stabilized nodal pressure
  hpc::device_vector<hpc::pressure<double>, point_index> K; // (tangent/effective) bulk modulus
  hpc::host_vector<hpc::device_vector<hpc::pressure<double>, node_index>, material_index> K_h; // (tangent/effective) bulk modulus at nodes
  hpc::device_vector<hpc::pressure<double>, point_index> G; // (tangent/effective) shear modulus
  hpc::device_vector<hpc::speed<double>, point_index> c; // sound speed / plane wave speed
  hpc::device_array_vector<hpc::force<double>, point_node_index> element_f; // (internal) force per element-node pair (contribution to a node's force by an element)
  hpc::device_array_vector<hpc::force<double>, point_node_index> support_f; // (internal) force per sopport-node pair (contribution to a node's force by a support)
  hpc::device_array_vector<hpc::force<double>, node_index> f; // nodal (internal) forces
  hpc::device_vector<hpc::density<double>, point_index> rho; // element density
  hpc::device_vector<hpc::specific_energy<double>, point_index> e; // element specific internal energy
  hpc::device_vector<hpc::energy_density_rate<double>, point_index> rho_e_dot; // time derivative of internal energy density
  hpc::device_vector<hpc::mass<double>, node_index> mass; // total lumped nodal mass
  hpc::host_vector<hpc::device_vector<hpc::mass<double>, node_index>, material_index> material_mass; // per-material lumped nodal mass
  hpc::device_array_vector<hpc::acceleration<double>, node_index> a; // nodal acceleration
  hpc::device_vector<hpc::length<double>, element_index> h_min; // minimum characteristic element length, used for stable time step
  hpc::device_vector<hpc::length<double>, element_index> h_art; // characteristic element length used for artificial viscosity
  hpc::device_vector<hpc::kinematic_viscosity<double>, point_index> nu_art; // artificial kinematic viscosity scalar
  hpc::device_vector<hpc::time<double>, point_index> element_dt; // stable time step of each element
  hpc::host_vector<hpc::device_vector<hpc::specific_energy<double>, node_index>, material_index> e_h; // nodal specific internal energy
  hpc::host_vector<hpc::device_vector<hpc::specific_energy_rate<double>, node_index>, material_index> e_h_dot; // time derivative of nodal specific internal energy
  hpc::host_vector<hpc::device_vector<hpc::density<double>, node_index>, material_index> rho_h; // nodal density
  hpc::host_vector<hpc::device_vector<dp_de_t, node_index>, material_index> dp_de_h; // nodal derivative of pressure with respect to energy, at constant density
  hpc::device_vector<material_index, element_index> material; // element material
  hpc::device_vector<material_set, node_index> nodal_materials; // nodal material set
  hpc::device_vector<hpc::dimensionless<double>, element_index> quality; // inverse element quality
  hpc::device_vector<hpc::length<double>, node_index> h_adapt; // desired edge length
  hpc::host_vector<hpc::device_vector<node_index, int>, material_index> node_sets;
  hpc::host_vector<hpc::device_vector<element_index, int>, material_index> element_sets;
  hpc::time<double> next_file_output_time;
  hpc::time<double> dt = 0.0;
  hpc::time<double> max_stable_dt;
  hpc::dimensionless<double> min_quality;

  hpc::device_array_vector<hpc::deformation_gradient<double>, point_index> Fp_total; // plastic deformation gradient since simulation start
  hpc::device_vector<hpc::temperature<double>, point_index> temp; // temperature
  hpc::device_vector<hpc::strain<double>, point_index> ep; // equivalent plastic strain
  hpc::device_vector<hpc::strain_rate<double>, point_index> ep_dot;  // rate of equivalent plastic strain

  // Hyper EP State dependent variables
  hpc::device_vector<hpc::dimensionless<double>, point_index> dp; // scalar damage
  hpc::device_vector<hpc::dimensionless<int>, point_index> localized; // localization flag
  hpc::host_vector<hpc::device_vector<hpc::dimensionless<double>, node_index>, material_index> ep_h;
};

class input;

void resize_state(input const& in, state& s);
void resize_state_meshless(input const& in, state& s);

}
