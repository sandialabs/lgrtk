#pragma once

#include <map>
#include <string>
#include <functional>

#include <hpc_vector3.hpp>
#include <lgr_domain.hpp>
#include <hpc_vector.hpp>
#include <hpc_range_sum.hpp>
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
  element_kind element{TETRAHEDRON};
  time_integrator_kind time_integrator = MIDPOINT_PREDICTOR_CORRECTOR;
  h_min_kind h_min = INBALL_DIAMETER;
  hpc::counting_range<material_index> materials;
  hpc::counting_range<material_index> boundaries;
  hpc::time<double> end_time{0.0};
  double CFL{0.9};
  bool use_constant_dt{false};
  bool use_contact{false};
  int minimum_support_size{4};
  hpc::time<double> constant_dt{0.0};
  int num_file_output_periods{0};
  int elements_along_x{0};
  hpc::length<double> x_domain_size{1.0};
  int elements_along_y{0};
  hpc::length<double> y_domain_size{1.0};
  int elements_along_z{0};
  hpc::length<double> z_domain_size{1.0};
  int otm_material_points_to_add_per_element{1};
  bool do_output{true};
  bool output_to_command_line{true};
  bool debug_output{false};
  hpc::host_vector<hpc::density<double>, material_index> rho0;
  hpc::host_vector<hpc::specific_energy<double>, material_index> e0;
  hpc::host_vector<bool, material_index> enable_neo_Hookean;
  hpc::host_vector<bool, material_index> enable_variational_J2;
  hpc::host_vector<hpc::pressure<double>, material_index> K0;
  hpc::host_vector<hpc::pressure<double>, material_index> G0;
  hpc::host_vector<bool, material_index> enable_ideal_gas;
  hpc::host_vector<double, material_index> gamma;
  hpc::host_vector<bool, material_index> enable_nodal_pressure;
  hpc::host_vector<bool, material_index> enable_nodal_energy;
  hpc::host_vector<bool, material_index> enable_p_prime;
  hpc::host_vector<double, material_index> c_tau;

  // OTM interpolation
  hpc::adimensional<double> otm_gamma{1.0}; // shape function locality parameter

  // Elasticity
  hpc::host_vector<hpc::pressure<double>, material_index> E;  // Young's modulus
  hpc::host_vector<double, material_index> nu;  // Poisson's ratio

  // Variational J2
  hpc::host_vector<hpc::pressure<double>, material_index> Y0;
  hpc::host_vector<hpc::adimensional<double>, material_index> n;
  hpc::host_vector<hpc::strain<double>, material_index> eps0;
  hpc::host_vector<hpc::pressure<double>, material_index> Svis0;
  hpc::host_vector<hpc::adimensional<double>, material_index> m;
  hpc::host_vector<hpc::strain_rate<double>, material_index> eps_dot0;

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
  bool enable_comptet_stabilization = false;
  hpc::length<double> max_node_neighbor_distance {1.0};
  hpc::length<double> max_point_neighbor_distance {1.0};
  std::function<
    void(hpc::counting_range<node_index> const,
        hpc::device_array_vector<hpc::position<double>, node_index> const&,
        hpc::device_array_vector<hpc::velocity<double>, node_index>*)> initial_v;
  std::vector<zero_acceleration_condition> zero_acceleration_conditions;
  std::function<void(hpc::device_array_vector<hpc::position<double>, node_index>*)> x_transform;
  std::function<
    void(hpc::counting_range<point_index> const,
        hpc::device_range_sum<point_node_index, point_index> const&,
        hpc::device_vector<node_index, point_node_index> const&,
        hpc::device_array_vector<hpc::position<double>, node_index> const&,
        hpc::device_array_vector<hpc::position<double>, point_index>&)> xp_transform;
  hpc::host_vector<std::unique_ptr<domain>, material_index> domains;
  input() = delete;
  input(material_index const material_count_in, material_index const boundary_count_in)
    :materials(material_count_in)
    ,boundaries(material_count_in, material_count_in + boundary_count_in)
    ,rho0(material_count_in)
    ,e0(material_count_in, double(0.0))
    ,enable_neo_Hookean(material_count_in, false)
    ,enable_variational_J2(material_count_in, false)
    ,K0(material_count_in)
    ,G0(material_count_in, double(0.0))
    ,enable_ideal_gas(material_count_in, false)
    ,gamma(material_count_in)
    ,enable_nodal_pressure(material_count_in, false)
    ,enable_nodal_energy(material_count_in, false)
    ,enable_p_prime(material_count_in, false)
    ,c_tau(material_count_in, 0.5)
    ,E(material_count_in)
    ,nu(material_count_in)
    ,Y0(material_count_in)
    ,n(material_count_in)
    ,eps0(material_count_in)
    ,Svis0(material_count_in)
    ,m(material_count_in)
    ,eps_dot0(material_count_in)
    ,allow_no_tension(material_count_in, true)
    ,allow_no_shear(material_count_in, false)
    ,set_stress_to_zero(material_count_in, false)
    ,D1(material_count_in)
    ,D2(material_count_in)
    ,D3(material_count_in)
    ,D4(material_count_in)
    ,D5(material_count_in)
    ,D6(material_count_in)
    ,D7(material_count_in)
    ,D8(material_count_in)
    ,DC(material_count_in)
    ,eps_f_min(material_count_in)
    ,domains(material_count_in + boundary_count_in)
  {}
};

}
