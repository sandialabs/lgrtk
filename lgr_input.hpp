#pragma once

#include <map>
#include <string>

#include <lgr_vector3.hpp>
#include <lgr_counting_range.hpp>
#include <lgr_domain.hpp>
#include <lgr_pinned_vector.hpp>
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
  pinned_memory_pool pinpool;
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
  pinned_vector<double, material_index> rho0;
  pinned_vector<double, material_index> e0;
  pinned_vector<bool, material_index> enable_neo_Hookean;
  pinned_vector<double, material_index> K0;
  pinned_vector<double, material_index> G0;
  pinned_vector<bool, material_index> enable_ideal_gas;
  pinned_vector<double, material_index> gamma;
  pinned_vector<bool, material_index> enable_nodal_pressure;
  pinned_vector<bool, material_index> enable_nodal_energy;
  double c_tau = 0.5;
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
    ,rho0(material_count_in, pinpool)
    ,e0(material_count_in, double(0.0), pinpool)
    ,enable_neo_Hookean(material_count_in, false, pinpool)
    ,K0(material_count_in, pinpool)
    ,G0(material_count_in, double(0.0), pinpool)
    ,enable_ideal_gas(material_count_in, false, pinpool)
    ,gamma(material_count_in, pinpool)
    ,enable_nodal_pressure(material_count_in, false, pinpool)
    ,enable_nodal_energy(material_count_in, false, pinpool)
    ,domains(material_count_in + boundary_count_in)
  {}
};

}
