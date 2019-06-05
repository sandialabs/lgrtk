#pragma once

#include <map>
#include <string>

#include <lgr_vector3.hpp>
#include <lgr_counting_range.hpp>
#include <lgr_domain.hpp>
#include <lgr_host_vector.hpp>

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
  vector3<double> axis;
};

class input {
  public:
  std::string name;
  element_kind element;
  time_integrator_kind time_integrator = MIDPOINT_PREDICTOR_CORRECTOR;
  h_min_kind h_min = INBALL_DIAMETER;
  counting_range<material_index> materials;
  counting_range<material_index> boundaries;
  double end_time;
  double CFL = 0.9;
  int num_file_outputs;
  int elements_along_x = 0;
  double x_domain_size = 1.0;
  int elements_along_y = 0;
  double y_domain_size = 1.0;
  int elements_along_z = 0;
  double z_domain_size = 1.0;
  bool output_to_command_line = true;
  host_vector<double, material_index> rho0;
  host_vector<double, material_index> e0;
  host_vector<bool, material_index> enable_neo_Hookean;
  host_vector<double, material_index> K0;
  host_vector<double, material_index> G0;
  host_vector<bool, material_index> enable_ideal_gas;
  host_vector<double, material_index> gamma;
  host_vector<bool, material_index> enable_nodal_pressure;
  host_vector<bool, material_index> enable_nodal_energy;
  host_vector<bool, material_index> enable_p_prime;
  host_vector<double, material_index> c_tau;
  bool enable_viscosity = false;
  double linear_artificial_viscosity = 0.0;
  double quadratic_artificial_viscosity = 0.0;
  bool enable_J_averaging = false;
  bool enable_rho_averaging = false;
  bool enable_e_averaging = false;
  bool enable_p_averaging = false;
  bool enable_adapt = false;
  std::function<
    void(counting_range<node_index> const,
        device_vector<vector3<double>, node_index> const&,
        device_vector<vector3<double>, node_index>*)> initial_v;
  std::vector<zero_acceleration_condition> zero_acceleration_conditions;
  std::function<void(device_vector<vector3<double>, node_index>*)> x_transform;
  host_vector<std::unique_ptr<domain>, material_index> domains;
  input() = delete;
  input(material_index const material_count_in, material_index const boundary_count_in)
    :materials(material_count_in)
    ,boundaries(material_count_in, material_count_in + boundary_count_in)
    ,rho0(material_count_in)
    ,e0(material_count_in, double(0.0))
    ,enable_neo_Hookean(material_count_in, false)
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
