#pragma once

#include <map>

#include <lgr_vector3.hpp>
#include <lgr_physics_types.hpp>
#include <lgr_counting_range.hpp>
#include <lgr_domain.hpp>

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
  std::string node_set_name;
  vector3<double> axis;
};

class input {
  public:
  pinned_memory_pool pinpool;
  std::string name;
  element_kind element;
  time_integrator_kind time_integrator = MIDPOINT_PREDICTOR_CORRECTOR;
  h_min_kind h_min = INBALL_DIAMETER;
  material_index material_count;
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
  bool enable_neo_Hookean = false;
  double K0;
  double G0 = 0.0;
  bool enable_ideal_gas = false;
  pinned_vector<double, material_index> gamma;
  bool enable_nodal_pressure = false;
  double c_tau = 0.5;
  bool enable_viscosity = false;
  double linear_artificial_viscosity = 0.0;
  double quadratic_artificial_viscosity = 0.0;
  bool enable_nodal_energy = false;
  bool enable_J_averaging = false;
  bool enable_rho_averaging = false;
  bool enable_e_averaging = false;
  std::function<
    void(counting_range<node_index> const,
        device_vector<vector3<double>, node_index> const&,
        device_vector<vector3<double>, node_index>*)> initial_v;
  std::vector<zero_acceleration_condition> zero_acceleration_conditions;
  std::map<std::string, std::unique_ptr<domain>> node_sets;
  std::function<void(device_vector<vector3<double>, node_index>*)> x_transform;
  std::vector<std::pair<material_index, std::unique_ptr<domain>>> material_domains;
  input() = delete;
  input(material_index material_count_in)
    :material_count(material_count_in)
    ,rho0(material_count_in, pinpool)
    ,e0(material_count_in, double(0.0), pinpool)
    ,gamma(material_count_in, pinpool)
  {}
};

}
