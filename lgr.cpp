#include <chrono>
#include <hpc_vector3.hpp>
#include <iostream>
#include <lgr_domain.hpp>
#include <lgr_input.hpp>
#include <lgr_physics.hpp>
#include <memory>

namespace lgr {

HPC_NOINLINE inline void
set_exponential_wave_v(
    hpc::counting_range<node_index> const                              nodes,
    hpc::device_array_vector<hpc::position<double>, node_index> const& x_vector,
    hpc::device_array_vector<hpc::velocity<double>, node_index>*       v_vector)
{
  auto const nodes_to_x = x_vector.cbegin();
  auto const nodes_to_v = v_vector->begin();
  auto       functor    = [=] HPC_DEVICE(node_index const node) {
    auto const x     = nodes_to_x[node].load();
    auto const d     = x(0) - 0.5;
    auto const v_x   = 1.0e-4 * std::exp(double(-(d * d) / (2 * (0.05 * 0.05))));
    nodes_to_v[node] = hpc::velocity<double>(v_x, 0.0, 0.0);
  };
  hpc::for_each(hpc::device_policy(), nodes, functor);
}

HPC_NOINLINE inline void
zero_v(
    hpc::counting_range<node_index> const /*nodes*/,
    hpc::device_array_vector<hpc::position<double>, node_index> const& /*x_vector*/,
    hpc::device_array_vector<hpc::velocity<double>, node_index>* v)
{
  hpc::fill(hpc::device_policy(), *v, hpc::velocity<double>::zero());
}

HPC_NOINLINE inline void
spin_v(
    hpc::counting_range<node_index> const                              nodes,
    hpc::device_array_vector<hpc::position<double>, node_index> const& x_vector,
    hpc::device_array_vector<hpc::velocity<double>, node_index>*       v_vector)
{
  auto const nodes_to_x = x_vector.cbegin();
  auto const nodes_to_v = v_vector->begin();
  auto       functor    = [=] HPC_DEVICE(node_index const node) {
    auto const x     = nodes_to_x[node].load();
    nodes_to_v[node] = 100.0 * hpc::velocity<double>(-(double(x(1)) - 0.5), (double(x(0)) - 0.5), 0.0);
  };
  hpc::for_each(hpc::device_policy(), nodes, functor);
}

HPC_NOINLINE void
elastic_wave();
void
elastic_wave()
{
  constexpr material_index body(0);
  constexpr material_index nmaterials(1);
  constexpr material_index x_boundary(1);
  constexpr material_index nboundaries(1);
  input                    in(nmaterials, nboundaries);
  in.name                          = "elastic_wave";
  in.element                       = BAR;
  in.end_time                      = 4.0e-3;
  in.num_file_output_periods       = 200;
  in.elements_along_x              = 1000;
  in.rho0[body]                    = 1000.0;
  in.enable_neo_Hookean[body]      = true;
  in.K0[body]                      = 1.0e9;
  in.G0[body]                      = 0.0;
  in.initial_v                     = set_exponential_wave_v;
  constexpr auto          x_axis   = hpc::vector3<double>::x_axis();
  static constexpr double eps      = 1.0e-10;
  auto                    x_domain = std::make_unique<union_domain>();
  x_domain->add(epsilon_around_plane_domain({x_axis, 0.0}, eps));
  x_domain->add(epsilon_around_plane_domain({x_axis, in.x_domain_size}, eps));
  in.domains[x_boundary] = std::move(x_domain);
  in.zero_acceleration_conditions.push_back({x_boundary, x_axis});
  // in.enable_nodal_pressure = true;
  // in.c_tau = 0.5;
  run(in);
}

HPC_NOINLINE void
gas_expansion();
void
gas_expansion()
{
  constexpr material_index gas(0);
  constexpr material_index nmaterials(1);
  constexpr material_index nboundaries(0);
  input                    in(nmaterials, nboundaries);
  in.name                    = "gas_expansion";
  in.element                 = BAR;
  in.end_time                = 10.0;
  in.num_file_output_periods = 100;
  in.elements_along_x        = 160;
  in.rho0[gas]               = 1.0;
  in.enable_ideal_gas[gas]   = true;
  in.gamma[gas]              = 1.4;
  in.e0[gas]                 = 1.0;
  in.initial_v               = zero_v;
  run(in);
}

HPC_NOINLINE void
spinning_square();
void
spinning_square()
{
  constexpr material_index body(0);
  constexpr material_index nmaterials(1);
  constexpr material_index nboundaries(0);
  input                    in(nmaterials, nboundaries);
  in.name                     = "spinning_square";
  in.element                  = TRIANGLE;
  in.end_time                 = 1.0e-2;
  in.num_file_output_periods  = 400;
  in.elements_along_x         = 1;
  in.x_domain_size            = 1.0;
  in.elements_along_y         = 1;
  in.y_domain_size            = 1.0;
  in.rho0[body]               = 1000.0;
  in.enable_neo_Hookean[body] = true;
  in.K0[body]                 = 200.0e9;
  in.G0[body]                 = 75.0e9;
  in.initial_v                = spin_v;
  run(in);
}

HPC_NOINLINE inline void
quadratic_in_x_v(
    hpc::counting_range<node_index> const                              nodes,
    hpc::device_array_vector<hpc::position<double>, node_index> const& x_vector,
    hpc::device_array_vector<hpc::velocity<double>, node_index>*       v_vector)
{
  auto const nodes_to_x = x_vector.cbegin();
  auto const nodes_to_v = v_vector->begin();
  auto       functor    = [=] HPC_DEVICE(node_index const node) {
    auto const x      = nodes_to_x[node].load();
    auto const norm_x = double(x(0) / 48.0);
    auto const v_y    = norm_x * norm_x;
    nodes_to_v[node]  = hpc::velocity<double>(0.0, v_y, 0.0);
  };
  hpc::for_each(hpc::device_policy(), nodes, functor);
}

HPC_NOINLINE inline void
Cooks_membrane_x(hpc::device_array_vector<hpc::position<double>, node_index>* x_vector)
{
  hpc::counting_range<node_index> const nodes(x_vector->size());
  auto const                            nodes_to_x = x_vector->begin();
  auto                                  functor    = [=] HPC_DEVICE(node_index const node) {
    auto const                  unit_x = nodes_to_x[node].load();
    hpc::position<double> const new_x(
        unit_x(0) * 48.0, unit_x(1) * 44.0 + unit_x(0) * 16.0 + double(unit_x(0) * (1.0 - unit_x(1)) * 28.0), 0.0);
    nodes_to_x[node] = new_x;
  };
  hpc::for_each(hpc::device_policy(), nodes, functor);
}

HPC_NOINLINE void
Cooks_membrane();
void
Cooks_membrane()
{
  constexpr material_index body(0);
  constexpr material_index nmaterials(1);
  constexpr material_index x_min(1);
  constexpr material_index nboundaries(1);
  input                    in(nmaterials, nboundaries);
  in.name                     = "Cooks_membrane";
  in.element                  = TRIANGLE;
  in.end_time                 = 40.0;
  in.num_file_output_periods  = 200;
  in.elements_along_x         = 8;
  in.x_domain_size            = 1.0;
  in.elements_along_y         = 8;
  in.y_domain_size            = 1.0;
  in.rho0[body]               = 1.0;
  in.enable_neo_Hookean[body] = true;
  in.K0[body]                 = 833333.0;
  in.G0[body]                 = 83.0;
  in.initial_v                = quadratic_in_x_v;
  static constexpr hpc::vector3<double> x_axis(1.0, 0.0, 0.0);
  static constexpr double               eps          = 1.0e-10;
  auto                                  x_min_domain = epsilon_around_plane_domain({x_axis, 0.0}, eps);
  in.domains[x_min]                                  = std::move(x_min_domain);
  static constexpr hpc::vector3<double> y_axis(0.0, 1.0, 0.0);
  in.zero_acceleration_conditions.push_back({x_min, x_axis});
  in.zero_acceleration_conditions.push_back({x_min, y_axis});
  in.x_transform                 = Cooks_membrane_x;
  in.enable_nodal_pressure[body] = true;
  in.c_tau[body]                 = 0.5;
  run(in);
}

HPC_NOINLINE void
swinging_plate();
void HPC_NOINLINE
swinging_plate()
{
  constexpr material_index body(0);
  constexpr material_index nmaterials(1);
  constexpr material_index x_min(1);
  constexpr material_index x_max(2);
  constexpr material_index y_min(3);
  constexpr material_index y_max(4);
  constexpr material_index nboundaries(4);
  input                    in(nmaterials, nboundaries);
  in.name                     = "swinging_plate";
  in.element                  = TRIANGLE;
  in.num_file_output_periods  = 200;
  in.elements_along_x         = 8;
  in.x_domain_size            = 2.0;
  in.elements_along_y         = 8;
  in.y_domain_size            = 2.0;
  double const rho            = 1.1e3;
  in.rho0[body]               = rho;
  in.enable_neo_Hookean[body] = true;
  double const nu             = 0.45;
  double const E              = 1.7e7;
  double const K              = E / (3.0 * (1.0 - 2.0 * nu));
  double const G              = E / (2.0 * (1.0 + nu));
  double const w              = (hpc::pi<double>() / 2.0) * std::sqrt((2.0 * G) / rho);
  in.end_time                 = 0.16;
  in.K0[body]                 = K;
  in.G0[body]                 = G;
  auto swinging_plate_v       = [=](hpc::counting_range<node_index> const                              nodes,
                              hpc::device_array_vector<hpc::position<double>, node_index> const& x_vector,
                              hpc::device_array_vector<hpc::velocity<double>, node_index>*       v_vector) {
    auto const   nodes_to_x = x_vector.cbegin();
    auto const   nodes_to_v = v_vector->begin();
    double const U0         = 0.001;
    auto         functor    = [=] HPC_DEVICE(node_index const node) {
      auto const x = nodes_to_x[node].load();
      auto const v =
          (U0 * w) *
          hpc::velocity<double>(
              -std::sin((hpc::pi<double>() * double(x(0))) / 2.0) * std::cos((hpc::pi<double>() * double(x(1))) / 2.0),
              std::cos((hpc::pi<double>() * double(x(0))) / 2.0) * std::sin((hpc::pi<double>() * double(x(1))) / 2.0),
              0.0);
      nodes_to_v[node] = v;
    };
    hpc::for_each(hpc::device_policy(), nodes, functor);
  };
  in.initial_v = swinging_plate_v;
  static constexpr hpc::vector3<double> x_axis(1.0, 0.0, 0.0);
  static constexpr hpc::vector3<double> y_axis(0.0, 1.0, 0.0);
  static constexpr double               eps = 1.0e-10;
  in.domains[x_min]                         = epsilon_around_plane_domain({x_axis, 0.0}, eps);
  in.domains[x_max]                         = epsilon_around_plane_domain({x_axis, in.x_domain_size}, eps);
  in.domains[y_min]                         = epsilon_around_plane_domain({y_axis, 0.0}, eps);
  in.domains[y_max]                         = epsilon_around_plane_domain({y_axis, in.y_domain_size}, eps);
  in.zero_acceleration_conditions.push_back({x_min, x_axis});
  in.zero_acceleration_conditions.push_back({y_min, y_axis});
  in.zero_acceleration_conditions.push_back({x_max, x_axis});
  in.zero_acceleration_conditions.push_back({y_max, y_axis});
  in.enable_nodal_pressure[body] = true;
  in.c_tau[body]                 = 0.5;
  run(in);
}

HPC_NOINLINE void
spinning_cube();
void
spinning_cube()
{
  constexpr material_index body(0);
  constexpr material_index nmaterials(1);
  constexpr material_index nboundaries(0);
  input                    in(nmaterials, nboundaries);
  in.name                     = "spinning_cube";
  in.element                  = TETRAHEDRON;
  in.end_time                 = 1.0e-2;
  in.num_file_output_periods  = 400;
  in.elements_along_x         = 1;
  in.x_domain_size            = 1.0;
  in.elements_along_y         = 1;
  in.y_domain_size            = 1.0;
  in.elements_along_z         = 1;
  in.z_domain_size            = 1.0;
  in.rho0[body]               = 7800.0;
  in.enable_neo_Hookean[body] = true;
  in.K0[body]                 = 200.0e9;
  in.G0[body]                 = 75.0e9;
  in.initial_v                = spin_v;
  in.CFL                      = 0.9;
  in.time_integrator          = VELOCITY_VERLET;
  run(in);
}

HPC_NOINLINE void
elastic_wave_2d();
void
elastic_wave_2d()
{
  constexpr material_index body(0);
  constexpr material_index nmaterials(1);
  constexpr material_index x_boundary(1);
  constexpr material_index y_boundary(2);
  constexpr material_index nboundaries(2);
  input                    in(nmaterials, nboundaries);
  in.name                     = "elastic_wave_2d";
  in.element                  = TRIANGLE;
  in.end_time                 = 2.0e-3;
  in.num_file_output_periods  = 100;
  in.elements_along_x         = 1000;
  in.elements_along_y         = 1;
  in.y_domain_size            = 1.0e-3;
  in.rho0[body]               = 1000.0;
  in.enable_neo_Hookean[body] = true;
  in.K0[body]                 = 1.0e9;
  in.G0[body]                 = 0.0;
  in.initial_v                = set_exponential_wave_v;
  static constexpr hpc::vector3<double> x_axis(1.0, 0.0, 0.0);
  static constexpr hpc::vector3<double> y_axis(0.0, 1.0, 0.0);
  static constexpr double               eps      = 1.0e-10;
  auto                                  x_domain = std::make_unique<union_domain>();
  x_domain->add(epsilon_around_plane_domain({x_axis, 0.0}, eps));
  x_domain->add(epsilon_around_plane_domain({x_axis, in.x_domain_size}, eps));
  in.domains[x_boundary] = std::move(x_domain);
  in.zero_acceleration_conditions.push_back({x_boundary, x_axis});
  auto y_domain = std::make_unique<union_domain>();
  y_domain->add(epsilon_around_plane_domain({y_axis, 0.0}, eps));
  y_domain->add(epsilon_around_plane_domain({y_axis, in.y_domain_size}, eps));
  in.domains[y_boundary] = std::move(y_domain);
  in.zero_acceleration_conditions.push_back({y_boundary, y_axis});
  run(in);
}

HPC_NOINLINE void
elastic_wave_3d();
void
elastic_wave_3d()
{
  constexpr material_index body(0);
  constexpr material_index nmaterials(1);
  constexpr material_index x_boundary(1);
  constexpr material_index y_boundary(2);
  constexpr material_index z_boundary(3);
  constexpr material_index nboundaries(3);
  input                    in(nmaterials, nboundaries);
  in.name                     = "elastic_wave_3d";
  in.element                  = TETRAHEDRON;
  in.end_time                 = 2.0e-3;
  in.num_file_output_periods  = 100;
  in.elements_along_x         = 1000;
  in.elements_along_y         = 1;
  in.y_domain_size            = 1.0e-3;
  in.elements_along_z         = 1;
  in.z_domain_size            = 1.0e-3;
  in.rho0[body]               = 1000.0;
  in.enable_neo_Hookean[body] = true;
  in.K0[body]                 = 1.0e9;
  in.G0[body]                 = 0.0;
  in.initial_v                = set_exponential_wave_v;
  static constexpr hpc::vector3<double> x_axis(1.0, 0.0, 0.0);
  static constexpr hpc::vector3<double> y_axis(0.0, 1.0, 0.0);
  static constexpr hpc::vector3<double> z_axis(0.0, 0.0, 1.0);
  static constexpr double               eps      = 1.0e-10;
  auto                                  x_domain = std::make_unique<union_domain>();
  x_domain->add(epsilon_around_plane_domain({x_axis, 0.0}, eps));
  x_domain->add(epsilon_around_plane_domain({x_axis, in.x_domain_size}, eps));
  in.domains[x_boundary] = std::move(x_domain);
  in.zero_acceleration_conditions.push_back({x_boundary, x_axis});
  auto y_domain = std::make_unique<union_domain>();
  y_domain->add(epsilon_around_plane_domain({y_axis, 0.0}, eps));
  y_domain->add(epsilon_around_plane_domain({y_axis, in.y_domain_size}, eps));
  in.domains[y_boundary] = std::move(y_domain);
  in.zero_acceleration_conditions.push_back({y_boundary, y_axis});
  auto z_domain = std::make_unique<union_domain>();
  z_domain->add(epsilon_around_plane_domain({z_axis, 0.0}, eps));
  z_domain->add(epsilon_around_plane_domain({z_axis, in.z_domain_size}, eps));
  in.domains[z_boundary] = std::move(z_domain);
  in.zero_acceleration_conditions.push_back({z_boundary, z_axis});
  run(in);
}

HPC_NOINLINE void
twisting_column_ep(
    double const end_time,
    bool const   plastic,
    bool const   output_to_command_line  = false,
    int const    num_file_output_periods = -1);
void
twisting_column_ep(
    double const end_time,
    bool const   plastic,
    bool const   output_to_command_line,
    int const    num_file_output_periods)
{
  constexpr material_index body(0);
  constexpr material_index nmaterials(1);
  constexpr material_index y_min(1);
  constexpr material_index nboundaries(1);
  input                    in(nmaterials, nboundaries);
  in.name = "twisting_column";
  if (plastic) in.name += "_ep";
  in.element  = TETRAHEDRON;
  in.end_time = end_time;
  if (num_file_output_periods == -1)
    in.num_file_output_periods = static_cast<int>(end_time / .001);
  else
    in.num_file_output_periods = num_file_output_periods;
  in.output_to_command_line      = output_to_command_line;
  in.elements_along_x            = 9;
  in.x_domain_size               = 1.0;
  in.elements_along_y            = 54;
  in.y_domain_size               = 6.0;
  in.elements_along_z            = 9;
  in.z_domain_size               = 1.0;
  auto const rho                 = hpc::density<double>(1.1e+03);
  auto const nu                  = hpc::adimensional<double>(0.499);
  auto const E                   = hpc::pressure<double>(17.0e06);
  auto const K                   = hpc::pressure<double>(E / (3.0 * (1.0 - 2.0 * nu)));
  auto const G                   = hpc::pressure<double>(E / (2.0 * (1.0 + nu)));
  auto const Y0                  = hpc::pressure<double>(1.0e+64);
  auto const n                   = hpc::adimensional<double>(4.0);
  auto const eps0                = hpc::strain<double>(1.0e-02);
  auto const Svis0               = hpc::pressure<double>(Y0);
  auto const m                   = hpc::adimensional<double>(2.0);
  auto const eps_dot0            = hpc::strain_rate<double>(1.0e-01);
  in.enable_variational_J2[body] = true;
  in.rho0[body]                  = rho;
  in.K0[body]                    = K;
  in.G0[body]                    = G;
  in.Y0[body]                    = Y0;
  in.n[body]                     = n;
  in.eps0[body]                  = eps0;
  in.Svis0[body]                 = Svis0;
  in.m[body]                     = m;
  in.eps_dot0[body]              = eps_dot0;
  in.CFL                         = 1.0;

  const double amplitude         = 100.0;
  auto         twisting_column_v = [=](hpc::counting_range<node_index> const                              nodes,
                               hpc::device_array_vector<hpc::position<double>, node_index> const& x_vector,
                               hpc::device_array_vector<hpc::velocity<double>, node_index>*       v_vector) {
    auto const nodes_to_x = x_vector.cbegin();
    auto const nodes_to_v = v_vector->begin();
    auto       functor    = [=] HPC_DEVICE(node_index const node) {
      auto const pos = nodes_to_x[node].load();
      auto const x   = double(pos(0));
      auto const y   = double(pos(1));
      auto const z   = double(pos(2));
      auto const v =
          amplitude * std::sin((hpc::pi<double>() / 12.0) * y) * hpc::velocity<double>((z - 0.5), 0.0, -(x - 0.5));
      nodes_to_v[node] = v;
    };
    hpc::for_each(hpc::device_policy(), nodes, functor);
  };
  in.initial_v = twisting_column_v;
  static constexpr hpc::vector3<double> x_axis(1.0, 0.0, 0.0);
  static constexpr hpc::vector3<double> y_axis(0.0, 1.0, 0.0);
  static constexpr hpc::vector3<double> z_axis(0.0, 0.0, 1.0);
  static constexpr double               eps = 1.0e-10;
  in.domains[y_min]                         = epsilon_around_plane_domain({y_axis, 0.0}, eps);
  in.zero_acceleration_conditions.push_back({y_min, x_axis});
  in.zero_acceleration_conditions.push_back({y_min, y_axis});
  in.zero_acceleration_conditions.push_back({y_min, z_axis});
  in.enable_nodal_pressure[body] = true;
  in.c_tau[body]                 = 0.5;
  in.CFL                         = 0.9;
  run(in);
}

HPC_NOINLINE void
swinging_cube(bool stabilize);
void
swinging_cube(bool stabilize)
{
  constexpr material_index body(0);
  constexpr material_index nmaterials(1);
  constexpr material_index x_min(1);
  constexpr material_index x_max(2);
  constexpr material_index y_min(3);
  constexpr material_index y_max(4);
  constexpr material_index z_min(5);
  constexpr material_index z_max(6);
  constexpr material_index nboundaries(6);
  input                    in(nmaterials, nboundaries);
  if (stabilize) {
    in.name = "stabilized_swinging_cube";
  } else {
    in.name = "swinging_cube";
  }
  in.element                  = TETRAHEDRON;
  in.num_file_output_periods  = 100;
  in.elements_along_x         = 8;
  in.x_domain_size            = 2.0;
  in.elements_along_y         = 8;
  in.y_domain_size            = 2.0;
  in.elements_along_z         = 8;
  in.z_domain_size            = 2.0;
  double const rho            = 1.1e3;
  in.rho0[body]               = rho;
  in.enable_neo_Hookean[body] = true;
  double const nu             = 0.45;
  double const E              = 1.7e7;
  double const K              = E / (3.0 * (1.0 - 2.0 * nu));
  double const G              = E / (2.0 * (1.0 + nu));
  double const w              = hpc::pi<double>() * std::sqrt((3.0 * G) / (4.0 * rho));
  in.end_time                 = 0.10;
  in.K0[body]                 = K;
  in.G0[body]                 = G;
  auto swinging_cube_v        = [=](hpc::counting_range<node_index> const                              nodes,
                             hpc::device_array_vector<hpc::position<double>, node_index> const& x_vector,
                             hpc::device_array_vector<hpc::velocity<double>, node_index>*       v_vector) {
    auto const   nodes_to_x = x_vector.cbegin();
    auto const   nodes_to_v = v_vector->begin();
    double const U0         = 5.0e-4;
    auto         functor    = [=] HPC_DEVICE(node_index const node) {
      constexpr double half_pi = hpc::pi<double>() / 2.0;
      auto const       pos     = nodes_to_x[node].load();
      auto const       x       = double(pos(0));
      auto const       y       = double(pos(1));
      auto const       z       = double(pos(2));
      auto const       v       = (U0 * w) * hpc::velocity<double>(
                                    -std::sin(half_pi * x) * std::cos(half_pi * y) * std::cos(half_pi * z),
                                    std::cos(half_pi * x) * std::sin(half_pi * y) * std::cos(half_pi * z),
                                    std::cos(half_pi * x) * std::cos(half_pi * y) * std::sin(half_pi * z));
      nodes_to_v[node] = v;
    };
    hpc::for_each(hpc::device_policy(), nodes, functor);
  };
  in.initial_v = swinging_cube_v;
  static constexpr hpc::vector3<double> x_axis(1.0, 0.0, 0.0);
  static constexpr hpc::vector3<double> y_axis(0.0, 1.0, 0.0);
  static constexpr hpc::vector3<double> z_axis(0.0, 0.0, 1.0);
  static constexpr double               eps = 1.0e-10;
  in.domains[x_min]                         = epsilon_around_plane_domain({x_axis, 0.0}, eps);
  in.domains[x_max]                         = epsilon_around_plane_domain({x_axis, in.x_domain_size}, eps);
  in.domains[y_min]                         = epsilon_around_plane_domain({y_axis, 0.0}, eps);
  in.domains[y_max]                         = epsilon_around_plane_domain({y_axis, in.y_domain_size}, eps);
  in.domains[z_min]                         = epsilon_around_plane_domain({z_axis, 0.0}, eps);
  in.domains[z_max]                         = epsilon_around_plane_domain({z_axis, in.z_domain_size}, eps);
  in.zero_acceleration_conditions.push_back({x_min, x_axis});
  in.zero_acceleration_conditions.push_back({x_max, x_axis});
  in.zero_acceleration_conditions.push_back({y_min, y_axis});
  in.zero_acceleration_conditions.push_back({y_max, y_axis});
  in.zero_acceleration_conditions.push_back({z_min, z_axis});
  in.zero_acceleration_conditions.push_back({z_max, z_axis});
  in.enable_nodal_pressure[body] = stabilize;
  in.c_tau[body]                 = 0.5;
  in.CFL                         = 0.45;
  run(in);
}

HPC_NOINLINE void
twisting_column();
void
twisting_column()
{
  constexpr material_index body(0);
  constexpr material_index nmaterials(1);
  constexpr material_index y_min(1);
  constexpr material_index nboundaries(1);
  input                    in(nmaterials, nboundaries);
  in.name                     = "twisting_column";
  in.element                  = TETRAHEDRON;
  in.end_time                 = 0.1;
  in.num_file_output_periods  = 100;
  in.elements_along_x         = 3;
  in.x_domain_size            = 1.0;
  in.elements_along_y         = 18;
  in.y_domain_size            = 6.0;
  in.elements_along_z         = 3;
  in.z_domain_size            = 1.0;
  double const rho            = 1.1e3;
  in.rho0[body]               = rho;
  in.enable_neo_Hookean[body] = true;
  double const nu             = 0.499;
  double const E              = 1.7e7;
  double const K              = E / (3.0 * (1.0 - 2.0 * nu));
  double const G              = E / (2.0 * (1.0 + nu));
  in.K0[body]                 = K;
  in.G0[body]                 = G;
  auto twisting_column_v      = [=](hpc::counting_range<node_index> const                              nodes,
                               hpc::device_array_vector<hpc::position<double>, node_index> const& x_vector,
                               hpc::device_array_vector<hpc::velocity<double>, node_index>*       v_vector) {
    auto const nodes_to_x = x_vector.cbegin();
    auto const nodes_to_v = v_vector->begin();
    auto       functor    = [=] HPC_DEVICE(node_index const node) {
      auto const pos = nodes_to_x[node].load();
      auto const x   = double(pos(0));
      auto const y   = double(pos(1));
      auto const z   = double(pos(2));
      auto const v =
          100.0 * std::sin((hpc::pi<double>() / 12.0) * y) * hpc::velocity<double>((z - 0.5), 0.0, -(x - 0.5));
      nodes_to_v[node] = v;
    };
    hpc::for_each(hpc::device_policy(), nodes, functor);
  };
  in.initial_v = twisting_column_v;
  static constexpr hpc::vector3<double> x_axis(1.0, 0.0, 0.0);
  static constexpr hpc::vector3<double> y_axis(0.0, 1.0, 0.0);
  static constexpr hpc::vector3<double> z_axis(0.0, 0.0, 1.0);
  static constexpr double               eps = 1.0e-10;
  in.domains[y_min]                         = epsilon_around_plane_domain({y_axis, 0.0}, eps);
  in.zero_acceleration_conditions.push_back({y_min, x_axis});
  in.zero_acceleration_conditions.push_back({y_min, y_axis});
  in.zero_acceleration_conditions.push_back({y_min, z_axis});
  in.enable_nodal_pressure[body] = false;
  in.c_tau[body]                 = 0.5;
  in.CFL                         = 0.9;
  run(in);
}

HPC_NOINLINE void
Noh_1D();
void
Noh_1D()
{
  constexpr material_index gas(0);
  constexpr material_index nmaterials(1);
  constexpr material_index x_min(1);
  constexpr material_index nboundaries(1);
  input                    in(nmaterials, nboundaries);
  in.name                    = "Noh_1D";
  in.element                 = BAR;
  in.end_time                = 0.6;
  in.num_file_output_periods = 60;
  in.elements_along_x        = 44;
  in.x_domain_size           = 1.1;
  in.rho0[gas]               = 1.0;
  in.enable_ideal_gas[gas]   = true;
  in.gamma[gas]              = 5.0 / 3.0;
  in.e0[gas]                 = 1.0e-4;
  auto inward_v              = [=](hpc::counting_range<node_index> const                              nodes,
                      hpc::device_array_vector<hpc::position<double>, node_index> const& x_vector,
                      hpc::device_array_vector<hpc::velocity<double>, node_index>*       v_vector) {
    auto const nodes_to_x = x_vector.cbegin();
    auto const nodes_to_v = v_vector->begin();
    auto       functor    = [=] HPC_DEVICE(node_index const node) {
      auto const x     = nodes_to_x[node].load();
      auto const n     = norm(x);
      auto const v     = (n == 0.0) ? hpc::velocity<double>::zero() : hpc::velocity<double>(-(x / n));
      nodes_to_v[node] = v;
    };
    hpc::for_each(hpc::device_policy(), nodes, functor);
  };
  in.initial_v = inward_v;
  static constexpr hpc::vector3<double> x_axis(1.0, 0.0, 0.0);
  static constexpr double               eps = 1.0e-10;
  in.domains[x_min]                         = epsilon_around_plane_domain({x_axis, 0.0}, eps);
  in.zero_acceleration_conditions.push_back({x_min, x_axis});
  in.enable_viscosity               = true;
  in.linear_artificial_viscosity    = 1.0;
  in.quadratic_artificial_viscosity = 1.0;
  in.enable_nodal_energy[gas]       = false;
  in.c_tau[gas]                     = 0.0;
  in.CFL                            = 0.9;
  run(in);
}

HPC_NOINLINE inline void
Noh_2D(bool nodal_energy, bool p_prime)
{
  constexpr material_index gas(0);
  constexpr material_index nmaterials(1);
  constexpr material_index x_min(1);
  constexpr material_index y_min(2);
  constexpr material_index nboundaries(2);
  input                    in(nmaterials, nboundaries);
  if (nodal_energy) {
    if (p_prime) {
      in.name = "Noh_2D_p_h_p_prime";
    } else {
      in.name = "Noh_2D_p_h";
    }
  } else {
    in.name = "Noh_2D";
  }
  in.element                 = TRIANGLE;
  in.end_time                = 0.6;
  in.num_file_output_periods = 60;
  in.elements_along_x        = 34;
  in.x_domain_size           = 0.85;
  in.elements_along_y        = 34;
  in.y_domain_size           = 0.85;
  in.rho0[gas]               = 1.0;
  in.enable_ideal_gas[gas]   = true;
  in.gamma[gas]              = 5.0 / 3.0;
  in.e0[gas]                 = 1.0e-14;
  auto inward_v              = [=](hpc::counting_range<node_index> const                              nodes,
                      hpc::device_array_vector<hpc::position<double>, node_index> const& x_vector,
                      hpc::device_array_vector<hpc::velocity<double>, node_index>*       v_vector) {
    auto const nodes_to_x = x_vector.cbegin();
    auto const nodes_to_v = v_vector->begin();
    auto       functor    = [=] HPC_DEVICE(node_index const node) {
      auto const x     = nodes_to_x[node].load();
      auto const n     = norm(x);
      auto const v     = (n == 0.0) ? hpc::velocity<double>::zero() : hpc::velocity<double>(-(x / n));
      nodes_to_v[node] = v;
    };
    hpc::for_each(hpc::device_policy(), nodes, functor);
  };
  in.initial_v = inward_v;
  static constexpr hpc::vector3<double> x_axis(1.0, 0.0, 0.0);
  static constexpr hpc::vector3<double> y_axis(0.0, 1.0, 0.0);
  static constexpr double               eps = 1.0e-10;
  in.domains[x_min]                         = epsilon_around_plane_domain({x_axis, 0.0}, eps);
  in.domains[y_min]                         = epsilon_around_plane_domain({y_axis, 0.0}, eps);
  in.zero_acceleration_conditions.push_back({x_min, x_axis});
  in.zero_acceleration_conditions.push_back({y_min, y_axis});
  in.enable_viscosity               = true;
  in.linear_artificial_viscosity    = 1.0;
  in.quadratic_artificial_viscosity = 0.5;
  in.enable_nodal_energy[gas]       = nodal_energy;
  in.enable_p_prime[gas]            = p_prime;
  in.c_tau[gas]                     = 1.0;
  run(in);
}

HPC_NOINLINE void
spinning_composite_cube();
void
spinning_composite_cube()
{
  constexpr material_index body(0);
  constexpr material_index nmaterials(1);
  constexpr material_index nboundaries(0);
  input                    in(nmaterials, nboundaries);
  in.name                     = "spinning_composite_cube";
  in.element                  = COMPOSITE_TETRAHEDRON;
  in.end_time                 = 1.0e-2;
  in.num_file_output_periods  = 400;
  in.elements_along_x         = 1;
  in.x_domain_size            = 1.0;
  in.elements_along_y         = 1;
  in.y_domain_size            = 1.0;
  in.elements_along_z         = 1;
  in.z_domain_size            = 1.0;
  in.rho0[body]               = 7800.0;
  in.enable_neo_Hookean[body] = true;
  in.K0[body]                 = 200.0e9;
  in.G0[body]                 = 75.0e9;
  in.initial_v                = spin_v;
  in.CFL                      = 0.9;
  in.time_integrator          = VELOCITY_VERLET;
  run(in);
}

HPC_NOINLINE void
twisting_composite_column();
void
twisting_composite_column()
{
  constexpr material_index body(0);
  constexpr material_index nmaterials(1);
  constexpr material_index y_min(1);
  constexpr material_index nboundaries(1);
  input                    in(nmaterials, nboundaries);
  in.name                     = "twisting_composite_column";
  in.element                  = COMPOSITE_TETRAHEDRON;
  in.end_time                 = 0.1;
  in.num_file_output_periods  = 100;
  in.elements_along_x         = 3;
  in.x_domain_size            = 1.0;
  in.elements_along_y         = 18;
  in.y_domain_size            = 6.0;
  in.elements_along_z         = 3;
  in.z_domain_size            = 1.0;
  double const rho            = 1.1e3;
  in.rho0[body]               = rho;
  in.enable_neo_Hookean[body] = true;
  double const nu             = 0.499;
  double const E              = 1.7e7;
  double const K              = E / (3.0 * (1.0 - 2.0 * nu));
  double const G              = E / (2.0 * (1.0 + nu));
  in.K0[body]                 = K;
  in.G0[body]                 = G;
  auto twisting_column_v      = [=](hpc::counting_range<node_index> const                              nodes,
                               hpc::device_array_vector<hpc::position<double>, node_index> const& x_vector,
                               hpc::device_array_vector<hpc::velocity<double>, node_index>*       v_vector) {
    auto const nodes_to_x = x_vector.cbegin();
    auto const nodes_to_v = v_vector->begin();
    auto       functor    = [=] HPC_DEVICE(node_index const node) {
      auto const x = hpc::vector3<double>(nodes_to_x[node].load());
      auto const v =
          100.0 * std::sin((hpc::pi<double>() / 12.0) * x(1)) * hpc::velocity<double>((x(2) - 0.5), 0.0, -(x(0) - 0.5));
      nodes_to_v[node] = v;
    };
    hpc::for_each(hpc::device_policy(), nodes, functor);
  };
  in.initial_v = twisting_column_v;
  static constexpr hpc::vector3<double> x_axis(1.0, 0.0, 0.0);
  static constexpr hpc::vector3<double> y_axis(0.0, 1.0, 0.0);
  static constexpr hpc::vector3<double> z_axis(0.0, 0.0, 1.0);
  static constexpr double               eps = 1.0e-10;
  in.domains[y_min]                         = epsilon_around_plane_domain({y_axis, 0.0}, eps);
  in.zero_acceleration_conditions.push_back({y_min, x_axis});
  in.zero_acceleration_conditions.push_back({y_min, y_axis});
  in.zero_acceleration_conditions.push_back({y_min, z_axis});
  in.enable_J_averaging = true;
  in.CFL                = 0.9;
  run(in);
}

HPC_NOINLINE void
twisting_composite_column_J2();
void
twisting_composite_column_J2()
{
  constexpr material_index body(0);
  constexpr material_index nmaterials(1);
  constexpr material_index y_min(1);
  constexpr material_index nboundaries(1);
  input                    in(nmaterials, nboundaries);
  in.name                    = "twisting_composite_column J2 plasticity";
  in.element                 = COMPOSITE_TETRAHEDRON;
  in.end_time                = 0.1;
  in.num_file_output_periods = 1000;
  in.elements_along_x        = 3;
  in.x_domain_size           = 1.0;
  in.elements_along_y        = 18;
  in.y_domain_size           = 6.0;
  in.elements_along_z        = 3;
  in.z_domain_size           = 1.0;
  double const rho{7.8e+03};
  double const nu{0.25};
  double const E{200.0e09};
  double const K{E / (3.0 * (1.0 - 2.0 * nu))};
  double const G{E / (2.0 * (1.0 + nu))};
  double const Y0{1.0e+09};
  double const n{4.0};
  double const eps0{1e-2};
  double const Svis0{Y0};
  double const m{2.0};
  double const eps_dot0{1e-1};
  in.enable_variational_J2[body] = true;
  in.rho0[body]                  = rho;
  in.K0[body]                    = K;
  in.G0[body]                    = G;
  in.Y0[body]                    = Y0;
  in.n[body]                     = n;
  in.eps0[body]                  = eps0;
  in.Svis0[body]                 = Svis0;
  in.m[body]                     = m;
  in.eps_dot0[body]              = eps_dot0;
  auto twisting_column_v         = [=](hpc::counting_range<node_index> const                              nodes,
                               hpc::device_array_vector<hpc::position<double>, node_index> const& x_vector,
                               hpc::device_array_vector<hpc::velocity<double>, node_index>*       v_vector) {
    auto const nodes_to_x = x_vector.cbegin();
    auto const nodes_to_v = v_vector->begin();
    auto       functor    = [=] HPC_DEVICE(node_index const node) {
      auto const x = hpc::vector3<double>(nodes_to_x[node].load());
      auto const v =
          100.0 * std::sin((hpc::pi<double>() / 12.0) * x(1)) * hpc::velocity<double>((x(2) - 0.5), 0.0, -(x(0) - 0.5));
      nodes_to_v[node] = v;
    };
    hpc::for_each(hpc::device_policy(), nodes, functor);
  };
  in.initial_v = twisting_column_v;
  static constexpr hpc::vector3<double> x_axis(1.0, 0.0, 0.0);
  static constexpr hpc::vector3<double> y_axis(0.0, 1.0, 0.0);
  static constexpr hpc::vector3<double> z_axis(0.0, 0.0, 1.0);
  static constexpr double               eps = 1.0e-10;
  in.domains[y_min]                         = epsilon_around_plane_domain({y_axis, 0.0}, eps);
  in.zero_acceleration_conditions.push_back({y_min, x_axis});
  in.zero_acceleration_conditions.push_back({y_min, y_axis});
  in.zero_acceleration_conditions.push_back({y_min, z_axis});
  in.enable_J_averaging = true;
  in.CFL                = 0.05;
  run(in);
}

HPC_NOINLINE void
flyer_target_stabilized_tet();
void
flyer_target_stabilized_tet()
{
  constexpr material_index flyer(0);
  constexpr material_index target(1);
  constexpr material_index num_materials(2);
  constexpr material_index num_boundaries(0);
  input                    in(num_materials, num_boundaries);
  std::string const        filename{"flyer-target.g"};
  auto const               flyer_radius = 0.2 * 0.0254;
  auto const               eps          = flyer_radius / 1000.0;

  auto flyer_v = [=](hpc::counting_range<node_index> const                              nodes,
                     hpc::device_array_vector<hpc::position<double>, node_index> const& x_vector,
                     hpc::device_array_vector<hpc::velocity<double>, node_index>*       v_vector) {
    auto const nodes_to_x = x_vector.cbegin();
    auto const nodes_to_v = v_vector->begin();
    auto       functor    = [=] HPC_DEVICE(node_index const node) {
      auto const x = hpc::vector3<double>(nodes_to_x[node].load());
      auto const r = std::sqrt(x(0) * x(0) + x(1) * x(1));
      auto       v = hpc::velocity<double>(0.0, 0.0, 0.0);
      if (r < flyer_radius + eps) {
        if (x(2) < -eps) v(2) = 2200.0;
        if (-eps <= x(2) && x(2) <= eps) v(2) = 242.0;
      }
      nodes_to_v[node] = v;
    };
    hpc::for_each(hpc::device_policy(), nodes, functor);
  };

  auto const rho      = hpc::density<double>(8.96e+03);
  auto const nu       = hpc::adimensional<double>(0.343);
  auto const E        = hpc::pressure<double>(110.0e09);
  auto const K        = hpc::pressure<double>(E / (3.0 * (1.0 - 2.0 * nu)));
  auto const G        = hpc::pressure<double>(E / (2.0 * (1.0 + nu)));
  auto const Y0       = hpc::pressure<double>(400.0e+06);
  auto const n        = hpc::adimensional<double>(1.0);
  auto const H0       = hpc::pressure<double>(100.0e6);
  auto const eps0     = hpc::strain<double>(Y0 / H0);
  auto const Svis0    = hpc::pressure<double>(0.0);
  auto const m        = hpc::adimensional<double>(1.0);
  auto const eps_dot0 = hpc::strain_rate<double>(1.0e-01);

  in.initial_v                      = flyer_v;
  in.name                           = "flyer-target";
  in.CFL                            = 0.1;
  in.element                        = TETRAHEDRON;
  in.end_time                       = 5.0e-06;
  in.num_file_output_periods        = 50;
  in.enable_J_averaging             = false;
  in.enable_p_averaging             = false;
  in.enable_viscosity               = true;
  in.linear_artificial_viscosity    = 2.0;
  in.quadratic_artificial_viscosity = 2.0;

  in.enable_p_prime[flyer]        = false;
  in.enable_nodal_pressure[flyer] = true;
  in.use_global_tau[flyer]        = true;
  in.c_tau[flyer]                 = 1.0;
  in.c_v[flyer]                   = 1.0;
  in.c_p[flyer]                   = 0.0;
  in.enable_variational_J2[flyer] = true;
  in.rho0[flyer]                  = rho;
  in.K0[flyer]                    = K;
  in.G0[flyer]                    = G;
  in.Y0[flyer]                    = Y0;
  in.n[flyer]                     = n;
  in.eps0[flyer]                  = eps0;
  in.Svis0[flyer]                 = Svis0;
  in.m[flyer]                     = m;
  in.eps_dot0[flyer]              = eps_dot0;

  in.enable_p_prime[target]        = false;
  in.enable_nodal_pressure[target] = true;
  in.use_global_tau[target]        = true;
  in.c_tau[target]                 = 1.0;
  in.c_v[target]                   = 1.0;
  in.c_p[target]                   = 0.0;
  in.enable_variational_J2[target] = true;
  in.rho0[target]                  = rho;
  in.K0[target]                    = K;
  in.G0[target]                    = G;
  in.Y0[target]                    = Y0;
  in.n[target]                     = n;
  in.eps0[target]                  = eps0;
  in.Svis0[target]                 = Svis0;
  in.m[target]                     = m;
  in.eps_dot0[target]              = eps_dot0;

  run(in, filename);
}

HPC_NOINLINE void
flyer_target_composite_tet();
void
flyer_target_composite_tet()
{
  constexpr material_index flyer(0);
  constexpr material_index target(1);
  constexpr material_index num_materials(2);
  constexpr material_index num_boundaries(0);
  input                    in(num_materials, num_boundaries);
  std::string const        filename{"flyer-target.g"};
  auto const               flyer_radius = 0.2 * 0.0254;
  auto const               eps          = flyer_radius / 1000.0;

  auto flyer_v = [=](hpc::counting_range<node_index> const                              nodes,
                     hpc::device_array_vector<hpc::position<double>, node_index> const& x_vector,
                     hpc::device_array_vector<hpc::velocity<double>, node_index>*       v_vector) {
    auto const nodes_to_x = x_vector.cbegin();
    auto const nodes_to_v = v_vector->begin();
    auto       functor    = [=] HPC_DEVICE(node_index const node) {
      auto const x = hpc::vector3<double>(nodes_to_x[node].load());
      auto const r = std::sqrt(x(0) * x(0) + x(1) * x(1));
      auto       v = hpc::velocity<double>(0.0, 0.0, 0.0);
      if (r < flyer_radius + eps) {
        if (x(2) < -eps) v(2) = 2200.0;
        if (-eps <= x(2) && x(2) <= eps) v(2) = 242.0;
      }
      nodes_to_v[node] = v;
    };
    hpc::for_each(hpc::device_policy(), nodes, functor);
  };

  auto const rho      = hpc::density<double>(8.96e+03);
  auto const nu       = hpc::adimensional<double>(0.343);
  auto const E        = hpc::pressure<double>(110.0e09);
  auto const K        = hpc::pressure<double>(E / (3.0 * (1.0 - 2.0 * nu)));
  auto const G        = hpc::pressure<double>(E / (2.0 * (1.0 + nu)));
  auto const Y0       = hpc::pressure<double>(400.0e+06);
  auto const n        = hpc::adimensional<double>(1.0);
  auto const H0       = hpc::pressure<double>(100.0e6);
  auto const eps0     = hpc::strain<double>(Y0 / H0);
  auto const Svis0    = hpc::pressure<double>(0.0);
  auto const m        = hpc::adimensional<double>(1.0);
  auto const eps_dot0 = hpc::strain_rate<double>(1.0e-01);

  in.initial_v                      = flyer_v;
  in.name                           = "flyer-target-ct";
  in.CFL                            = 0.1;
  in.element                        = COMPOSITE_TETRAHEDRON;
  in.end_time                       = 5.0e-06;
  in.num_file_output_periods        = 50;
  in.enable_J_averaging             = false;
  in.enable_p_averaging             = false;
  in.enable_viscosity               = true;
  in.linear_artificial_viscosity    = 1.0;
  in.quadratic_artificial_viscosity = 1.0;

  in.enable_p_prime[flyer]        = false;
  in.enable_nodal_pressure[flyer] = false;
  in.use_global_tau[flyer]        = true;
  in.c_tau[flyer]                 = 0.0;
  in.c_v[flyer]                   = 0.0;
  in.c_p[flyer]                   = 0.0;
  in.enable_variational_J2[flyer] = true;
  in.rho0[flyer]                  = rho;
  in.K0[flyer]                    = K;
  in.G0[flyer]                    = G;
  in.Y0[flyer]                    = Y0;
  in.n[flyer]                     = n;
  in.eps0[flyer]                  = eps0;
  in.Svis0[flyer]                 = Svis0;
  in.m[flyer]                     = m;
  in.eps_dot0[flyer]              = eps_dot0;

  in.enable_p_prime[target]        = false;
  in.enable_nodal_pressure[target] = false;
  in.use_global_tau[target]        = true;
  in.c_tau[target]                 = 0.0;
  in.c_v[target]                   = 0.0;
  in.c_p[target]                   = 0.0;
  in.enable_variational_J2[target] = true;
  in.rho0[target]                  = rho;
  in.K0[target]                    = K;
  in.G0[target]                    = G;
  in.Y0[target]                    = Y0;
  in.n[target]                     = n;
  in.eps0[target]                  = eps0;
  in.Svis0[target]                 = Svis0;
  in.m[target]                     = m;
  in.eps_dot0[target]              = eps_dot0;

  run(in, filename);
}

HPC_NOINLINE void
rmi_one_wave_stabilized_tet();
void
rmi_one_wave_stabilized_tet()
{
  constexpr material_index num_materials(2);
  constexpr material_index num_boundaries(4);
  constexpr material_index flyer(0);
  constexpr material_index target(1);
  constexpr material_index x_min(2);
  constexpr material_index x_max(3);
  constexpr material_index y_min(4);
  constexpr material_index y_max(5);
  input                    in(num_materials, num_boundaries);
  std::string const        filename{"rmi-one-wave.g"};
  auto const               flyer_radius = 0.2 * 0.0254;
  auto const               eps          = flyer_radius / 1000.0;

  auto flyer_v = [=](hpc::counting_range<node_index> const                              nodes,
                     hpc::device_array_vector<hpc::position<double>, node_index> const& x_vector,
                     hpc::device_array_vector<hpc::velocity<double>, node_index>*       v_vector) {
    auto const nodes_to_x = x_vector.cbegin();
    auto const nodes_to_v = v_vector->begin();
    auto       functor    = [=] HPC_DEVICE(node_index const node) {
      auto const x     = hpc::vector3<double>(nodes_to_x[node].load());
      auto const pos   = x(2);
      auto const s     = pos > eps ? 0.0 : (pos < -eps ? 2200.0 : 2200.0);
      auto       v     = hpc::velocity<double>(0.0, 0.0, s);
      nodes_to_v[node] = v;
    };
    hpc::for_each(hpc::device_policy(), nodes, functor);
  };

  auto const rho      = hpc::density<double>(8.93e+03);
  auto const nu       = hpc::adimensional<double>(0.343);
  auto const E        = hpc::pressure<double>(130.6e09);
  auto const K        = hpc::pressure<double>(E / (3.0 * (1.0 - 2.0 * nu)));
  auto const G        = hpc::pressure<double>(E / (2.0 * (1.0 + nu)));
  auto const Y0       = hpc::pressure<double>(89.7e+06);
  auto const n        = hpc::adimensional<double>(32.0);
  auto const H0       = hpc::pressure<double>(100.0e6);
  auto const eps0     = hpc::strain<double>(Y0 / H0);
  auto const Svis0    = hpc::pressure<double>(0.0);
  auto const m        = hpc::adimensional<double>(1.0);
  auto const eps_dot0 = hpc::strain_rate<double>(1.0e-01);
  auto const gamma    = hpc::adimensional<double>(1.99);
  auto const s        = hpc::adimensional<double>(1.489);
  auto const e0       = hpc::specific_energy<double>(0.0);

  static constexpr hpc::vector3<double> x_axis(1.0, 0.0, 0.0);
  static constexpr hpc::vector3<double> y_axis(0.0, 1.0, 0.0);
  static constexpr hpc::vector3<double> z_axis(0.0, 0.0, 1.0);

  auto const target_length = hpc::length<double>(0.001);
  auto const target_width  = hpc::length<double>(0.001 / 32.0);

  in.domains[x_min] = epsilon_around_plane_domain({x_axis, -target_length / 2.0}, eps);
  in.domains[x_max] = epsilon_around_plane_domain({x_axis, target_length / 2.0}, eps);
  in.domains[y_min] = epsilon_around_plane_domain({y_axis, -target_width / 2.0}, eps);
  in.domains[y_max] = epsilon_around_plane_domain({y_axis, target_width / 2.0}, eps);

  in.zero_acceleration_conditions.push_back({x_min, x_axis});
  in.zero_acceleration_conditions.push_back({x_max, x_axis});
  in.zero_acceleration_conditions.push_back({y_min, y_axis});
  in.zero_acceleration_conditions.push_back({y_max, y_axis});

  in.initial_v                      = flyer_v;
  in.name                           = "rmi-one-wave";
  in.CFL                            = 0.1;
  in.element                        = TETRAHEDRON;
  in.end_time                       = 10.0e-06;
  in.num_file_output_periods        = 100;
  in.enable_J_averaging             = false;
  in.enable_p_averaging             = false;
  in.enable_viscosity               = true;
  in.linear_artificial_viscosity    = 0.5;
  in.quadratic_artificial_viscosity = 2.0;
  in.enable_adapt                   = false;

  in.enable_nodal_energy[flyer]      = true;
  in.enable_Mie_Gruneisen_eos[flyer] = true;
  in.enable_p_prime[flyer]           = false;
  in.enable_nodal_pressure[flyer]    = true;
  in.use_global_tau[flyer]           = true;
  in.c_tau[flyer]                    = 1.0;
  in.c_v[flyer]                      = 1.0;
  in.c_p[flyer]                      = 0.0;
  in.enable_variational_J2[flyer]    = true;
  in.rho0[flyer]                     = rho;
  in.K0[flyer]                       = K;
  in.G0[flyer]                       = G;
  in.Y0[flyer]                       = Y0;
  in.n[flyer]                        = n;
  in.eps0[flyer]                     = eps0;
  in.Svis0[flyer]                    = Svis0;
  in.m[flyer]                        = m;
  in.eps_dot0[flyer]                 = eps_dot0;
  in.gamma[flyer]                    = gamma;
  in.s[flyer]                        = s;
  in.e0[flyer]                       = e0;

  in.enable_nodal_energy[target]      = true;
  in.enable_Mie_Gruneisen_eos[target] = true;
  in.enable_p_prime[target]           = false;
  in.enable_nodal_pressure[target]    = true;
  in.use_global_tau[target]           = true;
  in.c_tau[target]                    = 1.0;
  in.c_v[target]                      = 1.0;
  in.c_p[target]                      = 0.0;
  in.enable_variational_J2[target]    = true;
  in.rho0[target]                     = rho;
  in.K0[target]                       = K;
  in.G0[target]                       = G;
  in.Y0[target]                       = Y0;
  in.n[target]                        = n;
  in.eps0[target]                     = eps0;
  in.Svis0[target]                    = Svis0;
  in.m[target]                        = m;
  in.eps_dot0[target]                 = eps_dot0;
  in.gamma[target]                    = gamma;
  in.s[target]                        = s;
  in.e0[target]                       = e0;

  run(in, filename);
}

HPC_NOINLINE void
rmi_one_wave_composite_tet();
void
rmi_one_wave_composite_tet()
{
  constexpr material_index num_materials(2);
  constexpr material_index num_boundaries(4);
  constexpr material_index flyer(0);
  constexpr material_index target(1);
  constexpr material_index x_min(2);
  constexpr material_index x_max(3);
  constexpr material_index y_min(4);
  constexpr material_index y_max(5);
  input                    in(num_materials, num_boundaries);
  std::string const        filename{"rmi-one-wave-ct.g"};
  auto const               flyer_radius = 0.2 * 0.0254;
  auto const               eps          = flyer_radius / 1000.0;

  auto flyer_v = [=](hpc::counting_range<node_index> const                              nodes,
                     hpc::device_array_vector<hpc::position<double>, node_index> const& x_vector,
                     hpc::device_array_vector<hpc::velocity<double>, node_index>*       v_vector) {
    auto const nodes_to_x = x_vector.cbegin();
    auto const nodes_to_v = v_vector->begin();
    auto       functor    = [=] HPC_DEVICE(node_index const node) {
      auto const x     = hpc::vector3<double>(nodes_to_x[node].load());
      auto const pos   = x(2);
      auto const s     = pos > eps ? 0.0 : (pos < -eps ? 1045.0 : 1045.0);
      auto       v     = hpc::velocity<double>(0.0, 0.0, s);
      nodes_to_v[node] = v;
    };
    hpc::for_each(hpc::device_policy(), nodes, functor);
  };

  auto const rho      = hpc::density<double>(8.96e+03);
  auto const nu       = hpc::adimensional<double>(0.343);
  auto const E        = hpc::pressure<double>(110.0e09);
  auto const K        = hpc::pressure<double>(E / (3.0 * (1.0 - 2.0 * nu)));
  auto const G        = hpc::pressure<double>(E / (2.0 * (1.0 + nu)));
  auto const Y0       = hpc::pressure<double>(400.0e+06);
  auto const n        = hpc::adimensional<double>(32.0);
  auto const H0       = hpc::pressure<double>(100.0e6);
  auto const eps0     = hpc::strain<double>(Y0 / H0);
  auto const Svis0    = hpc::pressure<double>(0.0);
  auto const m        = hpc::adimensional<double>(1.0);
  auto const eps_dot0 = hpc::strain_rate<double>(1.0e-01);

  static constexpr hpc::vector3<double> x_axis(1.0, 0.0, 0.0);
  static constexpr hpc::vector3<double> y_axis(0.0, 1.0, 0.0);
  static constexpr hpc::vector3<double> z_axis(0.0, 0.0, 1.0);

  auto const target_length = hpc::length<double>(0.001);
  auto const target_width  = hpc::length<double>(0.001);

  in.domains[x_min] = epsilon_around_plane_domain({x_axis, -target_length / 2.0}, eps);
  in.domains[x_max] = epsilon_around_plane_domain({x_axis, target_length / 2.0}, eps);
  in.domains[y_min] = epsilon_around_plane_domain({y_axis, -target_width / 2.0}, eps);
  in.domains[y_max] = epsilon_around_plane_domain({y_axis, target_width / 2.0}, eps);

  in.zero_displacement_conditions.push_back({x_min, x_axis});
  in.zero_displacement_conditions.push_back({x_max, x_axis});
  in.zero_displacement_conditions.push_back({y_min, y_axis});
  in.zero_displacement_conditions.push_back({y_max, y_axis});

  in.initial_v                      = flyer_v;
  in.name                           = "rmi-one-wave-ct";
  in.CFL                            = 0.1;
  in.element                        = COMPOSITE_TETRAHEDRON;
  in.end_time                       = 5.0e-06;
  in.num_file_output_periods        = 50;
  in.enable_J_averaging             = false;
  in.enable_p_averaging             = false;
  in.enable_comptet_stabilization   = true;
  in.enable_viscosity               = true;
  in.linear_artificial_viscosity    = 0.5;
  in.quadratic_artificial_viscosity = 2.0;
  in.enable_adapt                   = false;

  in.enable_p_prime[flyer]        = false;
  in.enable_nodal_pressure[flyer] = false;
  in.use_global_tau[flyer]        = true;
  in.c_tau[flyer]                 = 0.0;
  in.c_v[flyer]                   = 0.0;
  in.c_p[flyer]                   = 0.0;
  in.enable_variational_J2[flyer] = true;
  in.rho0[flyer]                  = rho;
  in.K0[flyer]                    = K;
  in.G0[flyer]                    = G;
  in.Y0[flyer]                    = Y0;
  in.n[flyer]                     = n;
  in.eps0[flyer]                  = eps0;
  in.Svis0[flyer]                 = Svis0;
  in.m[flyer]                     = m;
  in.eps_dot0[flyer]              = eps_dot0;

  in.enable_p_prime[target]        = false;
  in.enable_nodal_pressure[target] = false;
  in.use_global_tau[target]        = true;
  in.c_tau[target]                 = 0.0;
  in.c_v[target]                   = 0.0;
  in.c_p[target]                   = 0.0;
  in.enable_variational_J2[target] = true;
  in.rho0[target]                  = rho;
  in.K0[target]                    = K;
  in.G0[target]                    = G;
  in.Y0[target]                    = Y0;
  in.n[target]                     = n;
  in.eps0[target]                  = eps0;
  in.Svis0[target]                 = Svis0;
  in.m[target]                     = m;
  in.eps_dot0[target]              = eps_dot0;

  run(in, filename);
}

HPC_NOINLINE void
mg_eos_verify_stabilized_tet();
void
mg_eos_verify_stabilized_tet()
{
  constexpr material_index num_materials(1);
  constexpr material_index num_boundaries(4);
  constexpr material_index body(0);
  constexpr material_index y_min(1);
  constexpr material_index y_max(2);
  constexpr material_index z_min(3);
  constexpr material_index z_max(4);
  input                    in(num_materials, num_boundaries);
  std::string const        filename{"mg-eos-verify.g"};
  auto const               height = hpc::length<double>(1.0e-5);
  auto const               width  = hpc::length<double>(1.0e-5);
  auto const               eps    = height / 100.0;

  auto flyer_v = [=](hpc::counting_range<node_index> const                              nodes,
                     hpc::device_array_vector<hpc::position<double>, node_index> const& x_vector,
                     hpc::device_array_vector<hpc::velocity<double>, node_index>*       v_vector) {
    auto const nodes_to_x = x_vector.cbegin();
    auto const nodes_to_v = v_vector->begin();
    auto       functor    = [=] HPC_DEVICE(node_index const node) {
      auto const x     = hpc::vector3<double>(nodes_to_x[node].load());
      auto const pos   = x(0);
      auto const s     = pos > eps ? 0.0 : (pos < -eps ? 100.0 : 50.0);
      auto       v     = hpc::velocity<double>(s, 0.0, 0.0);
      nodes_to_v[node] = v;
    };
    hpc::for_each(hpc::device_policy(), nodes, functor);
  };

  auto const rho      = hpc::density<double>(8.93e+03);
  auto const nu       = hpc::adimensional<double>(0.343);
  auto const E        = hpc::pressure<double>(130.6e09);
  auto const K        = hpc::pressure<double>(E / (3.0 * (1.0 - 2.0 * nu)));
  auto const G        = hpc::pressure<double>(E / (2.0 * (1.0 + nu)));
  auto const Y0       = hpc::pressure<double>(89.7e+06);
  auto const n        = hpc::adimensional<double>(32.0);
  auto const H0       = hpc::pressure<double>(100.0e6);
  auto const eps0     = hpc::strain<double>(Y0 / H0);
  auto const Svis0    = hpc::pressure<double>(0.0);
  auto const m        = hpc::adimensional<double>(1.0);
  auto const eps_dot0 = hpc::strain_rate<double>(1.0e-01);
  auto const gamma    = hpc::adimensional<double>(1.99);
  auto const s        = hpc::adimensional<double>(1.489);
  auto const e0       = hpc::specific_energy<double>(0.0);

  static constexpr hpc::vector3<double> x_axis(1.0, 0.0, 0.0);
  static constexpr hpc::vector3<double> y_axis(0.0, 1.0, 0.0);
  static constexpr hpc::vector3<double> z_axis(0.0, 0.0, 1.0);

  in.domains[y_min] = epsilon_around_plane_domain({y_axis, -height / 2.0}, eps);
  in.domains[y_max] = epsilon_around_plane_domain({y_axis, height / 2.0}, eps);
  in.domains[z_min] = epsilon_around_plane_domain({z_axis, -width / 2.0}, eps);
  in.domains[z_max] = epsilon_around_plane_domain({z_axis, width / 2.0}, eps);

  in.zero_acceleration_conditions.push_back({y_min, y_axis});
  in.zero_acceleration_conditions.push_back({y_max, y_axis});
  in.zero_acceleration_conditions.push_back({z_min, z_axis});
  in.zero_acceleration_conditions.push_back({z_max, z_axis});

  in.initial_v                      = flyer_v;
  in.name                           = "mg-eos-verify";
  in.CFL                            = 0.1;
  in.element                        = TETRAHEDRON;
  in.end_time                       = 1.8e-06;
  in.num_file_output_periods        = 9;
  in.enable_J_averaging             = false;
  in.enable_p_averaging             = false;
  in.enable_viscosity               = true;
  in.linear_artificial_viscosity    = 0.5;
  in.quadratic_artificial_viscosity = 2.0;
  in.enable_adapt                   = false;

  in.enable_nodal_energy[body]      = true;
  in.enable_Mie_Gruneisen_eos[body] = true;
  in.enable_p_prime[body]           = false;
  in.enable_nodal_pressure[body]    = true;
  in.use_global_tau[body]           = true;
  in.c_tau[body]                    = 1.0;
  in.c_v[body]                      = 1.0;
  in.c_p[body]                      = 0.0;
  in.enable_variational_J2[body]    = true;
  in.rho0[body]                     = rho;
  in.K0[body]                       = K;
  in.G0[body]                       = G;
  in.Y0[body]                       = Y0;
  in.n[body]                        = n;
  in.eps0[body]                     = eps0;
  in.Svis0[body]                    = Svis0;
  in.m[body]                        = m;
  in.eps_dot0[body]                 = eps_dot0;
  in.gamma[body]                    = gamma;
  in.s[body]                        = s;
  in.e0[body]                       = e0;

  run(in, filename);
}

HPC_NOINLINE void
uniaxial_tension();
void
uniaxial_tension()
{
  constexpr material_index num_materials(1);
  constexpr material_index num_boundaries(4);
  constexpr material_index body(0);
  constexpr material_index x_min(1);
  constexpr material_index y_min(2);
  constexpr material_index z_min(3);
  constexpr material_index z_max(4);

  input in(num_materials, num_boundaries);

  auto const eps = 1.0e-03;
  auto const speed = hpc::speed<double>(1.0);

  auto flyer_v = [=](hpc::counting_range<node_index> const                              nodes,
                     hpc::device_array_vector<hpc::position<double>, node_index> const& x_vector,
                     hpc::device_array_vector<hpc::velocity<double>, node_index>*       v_vector) {
    auto const nodes_to_x = x_vector.cbegin();
    auto const nodes_to_v = v_vector->begin();
    auto       functor    = [=] HPC_DEVICE(node_index const node) {
      auto const x     = hpc::vector3<double>(nodes_to_x[node].load());
      auto const pos   = x(2);
      auto const s     = speed * pos;
      auto       v     = hpc::velocity<double>(0.0, 0.0, s);
      nodes_to_v[node] = v;
    };
    hpc::for_each(hpc::device_policy(), nodes, functor);
  };

  auto const rho      = hpc::density<double>(8.93e+03);
  auto const nu       = hpc::adimensional<double>(0.343);
  auto const E        = hpc::pressure<double>(130.6e09);
  auto const K        = hpc::pressure<double>(E / (3.0 * (1.0 - 2.0 * nu)));
  auto const G        = hpc::pressure<double>(E / (2.0 * (1.0 + nu)));
  auto const Y0       = hpc::pressure<double>(89.7e+06);
  auto const n        = hpc::adimensional<double>(32.0);
  auto const H0       = hpc::pressure<double>(100.0e6);
  auto const eps0     = hpc::strain<double>(Y0 / H0);
  auto const Svis0    = hpc::pressure<double>(0.0);
  auto const m        = hpc::adimensional<double>(1.0);
  auto const eps_dot0 = hpc::strain_rate<double>(1.0e-01);
  auto const gamma    = hpc::adimensional<double>(1.99);
  auto const s        = hpc::adimensional<double>(1.489);
  auto const e0       = hpc::specific_energy<double>(0.0);

  static constexpr hpc::vector3<double> x_axis(1.0, 0.0, 0.0);
  static constexpr hpc::vector3<double> y_axis(0.0, 1.0, 0.0);
  static constexpr hpc::vector3<double> z_axis(0.0, 0.0, 1.0);

  in.domains[x_min] = epsilon_around_plane_domain({x_axis, 0.0}, eps);
  in.domains[y_min] = epsilon_around_plane_domain({y_axis, 0.0}, eps);
  in.domains[z_min] = epsilon_around_plane_domain({z_axis, 0.0}, eps);
  in.domains[z_max] = epsilon_around_plane_domain({z_axis, 1.0}, eps);

  in.prescribed_velocity_conditions.push_back({x_min, x_axis, 0.0});
  in.prescribed_velocity_conditions.push_back({y_min, y_axis, 0.0});
  in.prescribed_velocity_conditions.push_back({z_min, z_axis, 0.0});
  in.prescribed_velocity_conditions.push_back({z_max, z_axis, speed});

  in.initial_v                      = flyer_v;
  in.name                           = "uniaxial-tension";
  in.CFL                            = 0.9;
  in.element                        = TETRAHEDRON;
  in.end_time                       = 1.0;
  in.num_file_output_periods        = 100;
  in.elements_along_x               = 1;
  in.x_domain_size                  = 1.0;
  in.elements_along_y               = 1;
  in.y_domain_size                  = 1.0;
  in.elements_along_z               = 1;
  in.z_domain_size                  = 1.0;
  in.enable_J_averaging             = false;
  in.enable_p_averaging             = false;
  in.enable_viscosity               = false;
  in.linear_artificial_viscosity    = 0.5;
  in.quadratic_artificial_viscosity = 2.0;
  in.enable_adapt                   = false;

  in.enable_nodal_energy[body]      = false;
  in.enable_Mie_Gruneisen_eos[body] = false;
  in.enable_p_prime[body]           = false;
  in.enable_nodal_pressure[body]    = false;
  in.use_global_tau[body]           = true;
  in.c_tau[body]                    = 1.0;
  in.c_v[body]                      = 1.0;
  in.c_p[body]                      = 0.0;
  in.enable_variational_J2[body]    = true;
  in.rho0[body]                     = rho;
  in.K0[body]                       = K;
  in.G0[body]                       = G;
  in.Y0[body]                       = Y0;
  in.n[body]                        = n;
  in.eps0[body]                     = eps0;
  in.Svis0[body]                    = Svis0;
  in.m[body]                        = m;
  in.eps_dot0[body]                 = eps_dot0;
  in.gamma[body]                    = gamma;
  in.s[body]                        = s;
  in.e0[body]                       = e0;

  run(in);
}

HPC_NOINLINE void
taylor_composite_tet();
void
taylor_composite_tet()
{
  constexpr material_index num_materials(1);
  constexpr material_index num_boundaries(1);
  constexpr material_index body(0);
  constexpr material_index z_max(1);
  input                    in(num_materials, num_boundaries);
  std::string const        filename{"cylinder-ct.g"};

  auto const_v = [=](hpc::counting_range<node_index> const nodes,
                     hpc::device_array_vector<hpc::position<double>, node_index> const&,
                     hpc::device_array_vector<hpc::velocity<double>, node_index>* v_vector) {
    auto const nodes_to_v = v_vector->begin();
    auto       functor    = [=] HPC_DEVICE(node_index const node) {
      auto v           = hpc::velocity<double>(0.0, 0.0, 227.0);
      nodes_to_v[node] = v;
    };
    hpc::for_each(hpc::device_policy(), nodes, functor);
  };

  auto const rho      = hpc::density<double>(8.96e+03);
  auto const nu       = hpc::adimensional<double>(0.343);
  auto const E        = hpc::pressure<double>(110.0e09);
  auto const K        = hpc::pressure<double>(E / (3.0 * (1.0 - 2.0 * nu)));
  auto const G        = hpc::pressure<double>(E / (2.0 * (1.0 + nu)));
  auto const Y0       = hpc::pressure<double>(400.0e+06);
  auto const n        = hpc::adimensional<double>(32.0);
  auto const H0       = hpc::pressure<double>(100.0e6);
  auto const eps0     = hpc::strain<double>(Y0 / H0);
  auto const Svis0    = hpc::pressure<double>(0.0);
  auto const m        = hpc::adimensional<double>(1.0);
  auto const eps_dot0 = hpc::strain_rate<double>(1.0e-01);

  auto const                            eps = 1.0e-06;
  static constexpr hpc::vector3<double> z_axis(0.0, 0.0, 1.0);
  in.domains[z_max] = epsilon_around_plane_domain({z_axis, 0.0}, eps);
  in.zero_displacement_conditions.push_back({z_max, z_axis});

  in.name                         = "taylor-composite-tet";
  in.CFL                          = 0.1;
  in.use_displacement_contact     = false;
  in.use_penalty_contact          = false;
  in.contact_penalty_coeff        = hpc::strain_rate_rate<double>(1.0e+14);
  in.element                      = COMPOSITE_TETRAHEDRON;
  in.end_time                     = 1.0e-04;
  in.num_file_output_periods      = 100;
  in.enable_J_averaging           = true;
  in.enable_p_averaging           = true;
  in.enable_comptet_stabilization = true;
  in.enable_p_prime[body]         = false;
  in.enable_nodal_pressure[body]  = false;
  in.enable_variational_J2[body]  = true;
  in.rho0[body]                   = rho;
  in.K0[body]                     = K;
  in.G0[body]                     = G;
  in.Y0[body]                     = Y0;
  in.n[body]                      = n;
  in.eps0[body]                   = eps0;
  in.Svis0[body]                  = Svis0;
  in.m[body]                      = m;
  in.eps_dot0[body]               = eps_dot0;
  in.initial_v                    = const_v;

  run(in, filename);
}

HPC_NOINLINE void
taylor_stabilized_tet();
void
taylor_stabilized_tet()
{
  constexpr material_index num_materials(1);
  constexpr material_index num_boundaries(1);
  constexpr material_index body(0);
  constexpr material_index z_max(1);
  input                    in(num_materials, num_boundaries);
  std::string const        filename{"cylinder.g"};

  auto const_v = [=](hpc::counting_range<node_index> const nodes,
                     hpc::device_array_vector<hpc::position<double>, node_index> const&,
                     hpc::device_array_vector<hpc::velocity<double>, node_index>* v_vector) {
    auto const nodes_to_v = v_vector->begin();
    auto       functor    = [=] HPC_DEVICE(node_index const node) {
      auto v           = hpc::velocity<double>(0.0, 0.0, 227.0);
      nodes_to_v[node] = v;
    };
    hpc::for_each(hpc::device_policy(), nodes, functor);
  };

  auto const rho      = hpc::density<double>(8.96e+03);
  auto const nu       = hpc::adimensional<double>(0.343);
  auto const E        = hpc::pressure<double>(110.0e09);
  auto const K        = hpc::pressure<double>(E / (3.0 * (1.0 - 2.0 * nu)));
  auto const G        = hpc::pressure<double>(E / (2.0 * (1.0 + nu)));
  auto const Y0       = hpc::pressure<double>(400.0e+06);
  auto const n        = hpc::adimensional<double>(32.0);
  auto const H0       = hpc::pressure<double>(100.0e6);
  auto const eps0     = hpc::strain<double>(Y0 / H0);
  auto const Svis0    = hpc::pressure<double>(0.0);
  auto const m        = hpc::adimensional<double>(1.0);
  auto const eps_dot0 = hpc::strain_rate<double>(1.0e-01);

  auto const                            eps = 1.0e-06;
  static constexpr hpc::vector3<double> z_axis(0.0, 0.0, 1.0);
  in.domains[z_max] = epsilon_around_plane_domain({z_axis, 0.0}, eps);
  in.zero_displacement_conditions.push_back({z_max, z_axis});

  in.name                        = "taylor_stabilized-tet";
  in.CFL                         = 0.1;
  in.use_displacement_contact    = false;
  in.use_penalty_contact         = false;
  in.contact_penalty_coeff       = hpc::strain_rate_rate<double>(1.0e+14);
  in.element                     = TETRAHEDRON;
  in.end_time                    = 1.0e-04;
  in.num_file_output_periods     = 100;
  in.enable_J_averaging          = false;
  in.enable_p_averaging          = false;
  in.enable_p_prime[body]        = false;
  in.enable_nodal_pressure[body] = true;
  in.use_global_tau[body]        = true;
  in.c_tau[body]                 = 3.0;
  in.c_v[body]                   = 1.0;
  in.c_p[body]                   = 0.0;
  in.enable_variational_J2[body] = true;
  in.enable_neo_Hookean[body]    = false;
  in.rho0[body]                  = rho;
  in.K0[body]                    = K;
  in.G0[body]                    = G;
  in.Y0[body]                    = Y0;
  in.n[body]                     = n;
  in.eps0[body]                  = eps0;
  in.Svis0[body]                 = Svis0;
  in.m[body]                     = m;
  in.eps_dot0[body]              = eps_dot0;
  in.initial_v                   = const_v;

  run(in, filename);
}

HPC_NOINLINE void
Noh_3D();
void
Noh_3D()
{
  constexpr material_index gas(0);
  constexpr material_index nmaterials(1);
  constexpr material_index x_min(1);
  constexpr material_index y_min(2);
  constexpr material_index z_min(3);
  constexpr material_index nboundaries(3);
  input                    in(nmaterials, nboundaries);
  in.name                    = "Noh_3D";
  in.element                 = TETRAHEDRON;
  in.end_time                = 0.6;
  in.num_file_output_periods = 10;
  in.elements_along_x        = 20;
  in.x_domain_size           = 0.9;
  in.elements_along_y        = 20;
  in.y_domain_size           = 0.9;
  in.elements_along_z        = 20;
  in.z_domain_size           = 0.9;
  in.rho0[gas]               = 1.0;
  in.enable_ideal_gas[gas]   = true;
  in.gamma[gas]              = 5.0 / 3.0;
  in.e0[gas]                 = 1.0e-14;
  auto inward_v              = [=](hpc::counting_range<node_index> const                              nodes,
                      hpc::device_array_vector<hpc::position<double>, node_index> const& x_vector,
                      hpc::device_array_vector<hpc::velocity<double>, node_index>*       v_vector) {
    auto const nodes_to_x = x_vector.cbegin();
    auto const nodes_to_v = v_vector->begin();
    auto       functor    = [=] HPC_DEVICE(node_index const node) {
      auto const x     = nodes_to_x[node].load();
      auto const n     = norm(x);
      auto const v     = (n == 0.0) ? hpc::velocity<double>::zero() : hpc::velocity<double>(-(x / n));
      nodes_to_v[node] = v;
    };
    hpc::for_each(hpc::device_policy(), nodes, functor);
  };
  in.initial_v = inward_v;
  static constexpr hpc::vector3<double> x_axis(1.0, 0.0, 0.0);
  static constexpr hpc::vector3<double> y_axis(0.0, 1.0, 0.0);
  static constexpr hpc::vector3<double> z_axis(0.0, 0.0, 1.0);
  static constexpr double               eps = 1.0e-10;
  in.domains[x_min]                         = epsilon_around_plane_domain({x_axis, 0.0}, eps);
  in.domains[y_min]                         = epsilon_around_plane_domain({y_axis, 0.0}, eps);
  in.domains[z_min]                         = epsilon_around_plane_domain({z_axis, 0.0}, eps);
  in.zero_acceleration_conditions.push_back({x_min, x_axis});
  in.zero_acceleration_conditions.push_back({y_min, y_axis});
  in.zero_acceleration_conditions.push_back({z_min, z_axis});
  in.enable_viscosity               = true;
  in.linear_artificial_viscosity    = 1.0;
  in.quadratic_artificial_viscosity = 0.1;
  in.enable_nodal_energy[gas]       = true;
  in.c_tau[gas]                     = 1.0;
  run(in);
}

HPC_NOINLINE void
composite_Noh_3D();
void
composite_Noh_3D()
{
  constexpr material_index gas(0);
  constexpr material_index nmaterials(1);
  constexpr material_index x_min(1);
  constexpr material_index y_min(2);
  constexpr material_index z_min(3);
  constexpr material_index nboundaries(3);
  input                    in(nmaterials, nboundaries);
  in.name                    = "composite_Noh_3D";
  in.element                 = COMPOSITE_TETRAHEDRON;
  in.end_time                = 0.6;
  in.num_file_output_periods = 10;
  in.elements_along_x        = 20;
  in.x_domain_size           = 0.9;
  in.elements_along_y        = 20;
  in.y_domain_size           = 0.9;
  in.elements_along_z        = 20;
  in.z_domain_size           = 0.9;
  in.rho0[gas]               = 1.0;
  in.enable_ideal_gas[gas]   = true;
  in.gamma[gas]              = 5.0 / 3.0;
  in.e0[gas]                 = 1.0e-14;
  auto inward_v              = [=](hpc::counting_range<node_index> const                              nodes,
                      hpc::device_array_vector<hpc::position<double>, node_index> const& x_vector,
                      hpc::device_array_vector<hpc::velocity<double>, node_index>*       v_vector) {
    auto const nodes_to_x = x_vector.cbegin();
    auto const nodes_to_v = v_vector->begin();
    auto       functor    = [=] HPC_DEVICE(node_index const node) {
      auto const x     = nodes_to_x[node].load();
      auto const n     = norm(x);
      auto const v     = (n == 0.0) ? hpc::velocity<double>::zero() : hpc::velocity<double>(-(x / n));
      nodes_to_v[node] = v;
    };
    hpc::for_each(hpc::device_policy(), nodes, functor);
  };
  in.initial_v = inward_v;
  static constexpr hpc::vector3<double> x_axis(1.0, 0.0, 0.0);
  static constexpr hpc::vector3<double> y_axis(0.0, 1.0, 0.0);
  static constexpr hpc::vector3<double> z_axis(0.0, 0.0, 1.0);
  static constexpr double               eps = 1.0e-10;
  in.domains[x_min]                         = epsilon_around_plane_domain({x_axis, 0.0}, eps);
  in.domains[y_min]                         = epsilon_around_plane_domain({y_axis, 0.0}, eps);
  in.domains[z_min]                         = epsilon_around_plane_domain({z_axis, 0.0}, eps);
  in.zero_acceleration_conditions.push_back({x_min, x_axis});
  in.zero_acceleration_conditions.push_back({y_min, y_axis});
  in.zero_acceleration_conditions.push_back({z_min, z_axis});
  in.enable_viscosity               = true;
  in.linear_artificial_viscosity    = 0.25;
  in.quadratic_artificial_viscosity = 0.5;
  in.enable_p_averaging             = false;
  in.enable_rho_averaging           = false;
  in.enable_e_averaging             = true;
  run(in);
}

HPC_NOINLINE void
Sod_1D();
void
Sod_1D()
{
  constexpr material_index left(0);
  constexpr material_index right(1);
  constexpr material_index nmaterials(2);
  constexpr material_index x_min(2);
  constexpr material_index x_max(3);
  constexpr material_index nboundaries(2);
  input                    in(nmaterials, nboundaries);
  in.name                    = "Sod_1D";
  in.element                 = BAR;
  in.end_time                = 0.14;
  in.num_file_output_periods = 14;
  in.elements_along_x        = 100;
  in.x_domain_size           = 1.0;
  in.rho0[left]              = 1.0;
  in.rho0[right]             = 0.125;
  in.enable_ideal_gas[left]  = true;
  in.enable_ideal_gas[right] = true;
  in.gamma[left]             = 1.4;
  in.gamma[right]            = 1.4;
  in.e0[left]                = 1.0 / ((1.4 - 1.0) * 1.0);
  in.e0[right]               = 0.1 / ((1.4 - 1.0) * 0.125);
  in.initial_v               = zero_v;
  static constexpr hpc::vector3<double> x_axis(1.0, 0.0, 0.0);
  static constexpr double               eps = 1.0e-10;
  in.domains[x_min]                         = epsilon_around_plane_domain({x_axis, 0.0}, eps);
  in.domains[x_max]                         = epsilon_around_plane_domain({x_axis, 0.0}, eps);
  in.zero_acceleration_conditions.push_back({x_min, x_axis});
  in.zero_acceleration_conditions.push_back({x_max, x_axis});
  in.enable_viscosity               = true;
  in.linear_artificial_viscosity    = 0.5;
  in.quadratic_artificial_viscosity = 0.125;
  auto right_domain                 = half_space_domain(plane{hpc::vector3<double>{1.0, 0.0, 0.0}, 0.5});
  auto left_domain                  = half_space_domain(plane{hpc::vector3<double>{-1.0, 0.0, 0.0}, -0.5});
  in.domains[left]                  = std::move(left_domain);
  in.domains[right]                 = std::move(right_domain);
  in.enable_nodal_energy[left]      = true;
  in.enable_nodal_energy[right]     = true;
  in.c_tau[left]                    = 0.0;
  in.c_tau[right]                   = 0.0;
  run(in);
}

HPC_NOINLINE void
triple_point();
void
triple_point()
{
  constexpr material_index left(0);
  constexpr material_index right_bottom(1);
  constexpr material_index right_top(2);
  constexpr material_index nmaterials(3);
  constexpr material_index x_min(3);
  constexpr material_index x_max(4);
  constexpr material_index y_min(5);
  constexpr material_index y_max(6);
  constexpr material_index nboundaries(4);
  input                    in(nmaterials, nboundaries);
  in.name                           = "triple_point";
  in.element                        = TRIANGLE;
  in.end_time                       = 6.0;
  in.num_file_output_periods        = 60;
  in.elements_along_x               = 56;
  in.x_domain_size                  = 7.0;
  in.elements_along_y               = 24;
  in.y_domain_size                  = 3.0;
  in.rho0[right_top]                = 0.1;
  in.rho0[right_bottom]             = 1.0;
  in.rho0[left]                     = 1.0;
  in.enable_ideal_gas[left]         = true;
  in.enable_ideal_gas[right_top]    = true;
  in.enable_ideal_gas[right_bottom] = true;
  in.gamma[right_top]               = 1.5;
  in.gamma[left]                    = 1.5;
  in.gamma[right_bottom]            = 1.4;
  in.e0[right_top]                  = 2.5;
  in.e0[right_bottom]               = 0.3125;
  in.e0[left]                       = 2.0;
  in.initial_v                      = zero_v;
  constexpr auto   x_axis           = hpc::vector3<double>::x_axis();
  constexpr auto   y_axis           = hpc::vector3<double>::y_axis();
  constexpr double eps              = 1.0e-10;
  in.domains[x_min]                 = epsilon_around_plane_domain({x_axis, 0.0}, eps);
  in.domains[x_max]                 = epsilon_around_plane_domain({x_axis, in.x_domain_size}, eps);
  in.domains[y_min]                 = epsilon_around_plane_domain({y_axis, 0.0}, eps);
  in.domains[y_max]                 = epsilon_around_plane_domain({y_axis, in.y_domain_size}, eps);
  in.zero_acceleration_conditions.push_back({x_min, x_axis});
  in.zero_acceleration_conditions.push_back({x_max, x_axis});
  in.zero_acceleration_conditions.push_back({y_min, y_axis});
  in.zero_acceleration_conditions.push_back({y_max, y_axis});
  auto left_domain                     = box_domain({0.0, 0.0, -eps}, {1.0, 3.0, eps});
  auto right_bottom_domain             = box_domain({1.0, 0.0, -eps}, {7.0, 1.5, eps});
  auto right_top_domain                = box_domain({1.0, 1.5, -eps}, {7.0, 3.0, eps});
  in.domains[left]                     = std::move(left_domain);
  in.domains[right_bottom]             = std::move(right_bottom_domain);
  in.domains[right_top]                = std::move(right_top_domain);
  in.enable_viscosity                  = true;
  in.linear_artificial_viscosity       = 0.5;
  in.enable_nodal_energy[left]         = false;
  in.enable_nodal_energy[right_bottom] = false;
  in.enable_nodal_energy[right_top]    = false;
  in.c_tau[left]                       = 1.0;
  in.c_tau[right_bottom]               = 1.0;
  in.c_tau[right_top]                  = 1.0;
  in.enable_adapt                      = true;
  run(in);
}

}  // namespace lgr

HPC_NOINLINE void
run_for_average();
void
run_for_average()
{
  for (auto plastic : {true, false}) {
    std::cout << "Starting simulations with plastic = " << ((plastic) ? "true" : "false") << "\n";
    auto const start = std::chrono::high_resolution_clock::now();
    int        n     = 1;
    for (; n <= 5; n++) {
      std::cout << "  Running n = " << n << "\n";
      lgr::twisting_column_ep(0.005, plastic, false, 0);
    }
    auto const stop           = std::chrono::high_resolution_clock::now();
    auto       total_duration = std::chrono::duration_cast<std::chrono::seconds>(stop - start);
    auto       avg_duration   = total_duration / n;
    std::cout << "  Running n = " << n << "\n";
    lgr::twisting_column_ep(0.005, plastic, true, 30);
    std::cout << "Finished simulations with plastic = " << ((plastic) ? "true" : "false")
              << " with an average simulation time of " << avg_duration.count() << " seconds.\n";
  }
}

int
main(int ac, char* av[])
{
  std::string const problem = ac > 1 ? av[1] : "";
  HPC_TRAP_FPE();
  if (problem == "composite_Noh_3D")
    lgr::composite_Noh_3D();
  else if (problem == "Cooks_membrane")
    lgr::Cooks_membrane();
  else if (problem == "elastic_wave")
    lgr::elastic_wave();
  else if (problem == "elastic_wave_2d")
    lgr::elastic_wave_2d();
  else if (problem == "elastic_wave_3d")
    lgr::elastic_wave_3d();
  else if (problem == "flyer_target_composite_tet")
    lgr::flyer_target_composite_tet();
  else if (problem == "flyer_target_stabilized_tet")
    lgr::flyer_target_stabilized_tet();
  else if (problem == "gas_expansion")
    lgr::gas_expansion();
  else if (problem == "Noh_1D")
    lgr::Noh_1D();
  else if (problem == "Noh_2D_0_0")
    lgr::Noh_2D(false, false);
  else if (problem == "Noh_2D_1_0")
    lgr::Noh_2D(true, false);
  else if (problem == "Noh_2D_1_1")
    lgr::Noh_2D(true, true);
  else if (problem == "Noh_3D")
    lgr::Noh_3D();
  else if (problem == "rmi_one_wave_composite_tet")
    lgr::rmi_one_wave_composite_tet();
  else if (problem == "rmi_one_wave_stabilized_tet")
    lgr::rmi_one_wave_stabilized_tet();
  else if (problem == "mg_eos_verify_stabilized_tet")
    lgr::mg_eos_verify_stabilized_tet();
  else if (problem == "uniaxial_tension")
    lgr::uniaxial_tension();
  else if (problem == "Sod_1D")
    lgr::Sod_1D();
  else if (problem == "spinning_composite_cube")
    lgr::spinning_composite_cube();
  else if (problem == "spinning_cube")
    lgr::spinning_cube();
  else if (problem == "spinning_square")
    lgr::spinning_square();
  else if (problem == "swinging_cube_0")
    lgr::swinging_cube(false);
  else if (problem == "swinging_cube_1")
    lgr::swinging_cube(true);
  else if (problem == "swinging_plate")
    lgr::swinging_plate();
  else if (problem == "taylor_composite_tet")
    lgr::taylor_composite_tet();
  else if (problem == "taylor_stabilized_tet")
    lgr::taylor_stabilized_tet();
  else if (problem == "triple_point")
    lgr::triple_point();
  else if (problem == "twisting_column_ep_0")
    lgr::twisting_column_ep(0.05, false);
  else if (problem == "twisting_column_ep_1")
    lgr::twisting_column_ep(0.05, true);
  else if (problem == "twisting_column")
    lgr::twisting_column();
  else if (problem == "twisting_composite_column_J2")
    lgr::twisting_composite_column_J2();
  else if (problem == "twisting_composite_column")
    lgr::twisting_composite_column();
  else {
    std::cout << "Unrecognized problem string.\n";
  }
  // run_for_average();
}
