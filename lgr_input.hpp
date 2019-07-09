#pragma once

#include <map>
#include <string>
#include <functional>

#include <hpc_vector3.hpp>
#include <lgr_domain.hpp>
#include <hpc_vector.hpp>
#include <hpc_dimensional.hpp>

namespace lgr {

enum element_kind {
  BAR,
  TRIANGLE,
  TETRAHEDRON,
  COMPOSITE_TETRAHEDRON,
};

enum time_integrator_kind {
  MIDPOINT_PREDICTOR_CORRECTOR,
  VELOCITY_VERLET,
};

enum h_min_kind {
  MINIMUM_HEIGHT,
  INBALL_DIAMETER,
};

class zero_acceleration_condition {
  public:
  material_index boundary;
  hpc::vector3<double> axis;
};

class input {
  public:
  std::string name;
  element_kind element;
  time_integrator_kind time_integrator = MIDPOINT_PREDICTOR_CORRECTOR;
  h_min_kind h_min = INBALL_DIAMETER;
  hpc::counting_range<material_index> materials;
  hpc::counting_range<material_index> boundaries;
  hpc::time<double> end_time;
  double CFL = 0.9;
  int num_file_outputs;
  int elements_along_x = 0;
  hpc::length<double> x_domain_size = 1.0;
  int elements_along_y = 0;
  hpc::length<double> y_domain_size = 1.0;
  int elements_along_z = 0;
  hpc::length<double> z_domain_size = 1.0;
  bool output_to_command_line = true;
  hpc::host_vector<hpc::density<double>, material_index> rho0;
  hpc::host_vector<hpc::specific_energy<double>, material_index> e0;
  hpc::host_vector<bool, material_index> enable_neo_Hookean;
  hpc::host_vector<bool, material_index> enable_hyper_ep;
  hpc::host_vector<hpc::pressure<double>, material_index> K0;
  hpc::host_vector<hpc::pressure<double>, material_index> G0;
  hpc::host_vector<bool, material_index> enable_ideal_gas;
  hpc::host_vector<double, material_index> gamma;
  hpc::host_vector<bool, material_index> enable_nodal_pressure;
  hpc::host_vector<bool, material_index> enable_nodal_energy;
  hpc::host_vector<bool, material_index> enable_p_prime;
  hpc::host_vector<double, material_index> c_tau;

  // Inputs for the hyper elastic-plastic model
  hpc::host_vector<hpc::pressure<double>, material_index> E;  // Young's modulus
  hpc::host_vector<double, material_index> Nu;  // Poisson's ratio

  // Plasticity
  hpc::host_vector<hpc::pressure<double>, material_index> A;  // Yield strength in shear
  hpc::host_vector<hpc::pressure<double>, material_index> B;  // hardenint modules
  hpc::host_vector<double, material_index> n;  // hardening exponent
  hpc::host_vector<double, material_index> C1;
  hpc::host_vector<double, material_index> C2;
  hpc::host_vector<double, material_index> C3;
  hpc::host_vector<double, material_index> C4;
  hpc::host_vector<double, material_index> ep_dot_0;

  // Damage
  hpc::host_vector<bool, material_index> allow_no_tension;
  hpc::host_vector<bool, material_index> allow_no_shear;
  hpc::host_vector<bool, material_index> set_stress_to_zero;
  hpc::host_vector<double, material_index> D1;
  hpc::host_vector<double, material_index> D2;
  hpc::host_vector<double, material_index> D3;
  hpc::host_vector<double, material_index> D4;
  hpc::host_vector<double, material_index> D5;
  hpc::host_vector<double, material_index> D6;
  hpc::host_vector<double, material_index> D7;
  hpc::host_vector<double, material_index> D8;
  hpc::host_vector<double, material_index> DC;
  hpc::host_vector<double, material_index> eps_f_min;

  bool enable_viscosity = false;
  double linear_artificial_viscosity = 0.0;
  double quadratic_artificial_viscosity = 0.0;
  bool enable_J_averaging = false;
  bool enable_rho_averaging = false;
  bool enable_e_averaging = false;
  bool enable_p_averaging = false;
  bool enable_adapt = false;
  std::function<
    void(hpc::counting_range<node_index> const,
        hpc::device_array_vector<hpc::position<double>, node_index> const&,
        hpc::device_array_vector<hpc::velocity<double>, node_index>*)> initial_v;
  std::vector<zero_acceleration_condition> zero_acceleration_conditions;
  std::function<void(hpc::device_array_vector<hpc::position<double>, node_index>*)> x_transform;
  hpc::host_vector<std::unique_ptr<domain>, material_index> domains;
  input() = delete;
  input(material_index const material_count_in, material_index const boundary_count_in)
    :materials(material_count_in)
    ,boundaries(material_count_in, material_count_in + boundary_count_in)
    ,rho0(material_count_in)
    ,e0(material_count_in, double(0.0))
    ,enable_neo_Hookean(material_count_in, false)
    ,enable_hyper_ep(material_count_in, false)
    ,K0(material_count_in)
    ,G0(material_count_in, double(0.0))
    ,enable_ideal_gas(material_count_in, false)
    ,gamma(material_count_in)
    ,enable_nodal_pressure(material_count_in, false)
    ,enable_nodal_energy(material_count_in, false)
    ,enable_p_prime(material_count_in, false)
    ,c_tau(material_count_in, 0.5)
    ,domains(material_count_in + boundary_count_in)
  {}
};

}
