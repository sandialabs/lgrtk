#include <memory>

#include <lgr_physics.hpp>
#include <lgr_for_each.hpp>
#include <lgr_fill.hpp>
#include <lgr_domain.hpp>
#include <lgr_input.hpp>

namespace lgr {

static void LGR_NOINLINE set_exponential_wave_v(
    counting_range<node_index> const nodes,
    hpc::device_array_vector<vector3<double>, node_index> const& x_vector,
    hpc::device_array_vector<vector3<double>, node_index>* v_vector) {
  auto const nodes_to_x = x_vector.cbegin();
  auto const nodes_to_v = v_vector->begin();
  auto functor = [=](node_index const node) {
    auto const x = nodes_to_x[node].load();
    auto const d = x(0) - 0.5;
    auto const v_x = 1.0e-4 * std::exp(-(d * d) / (2 * (0.05 * 0.05)));
    nodes_to_v[node] = vector3<double>(v_x, 0.0, 0.0);
  };
  lgr::for_each(nodes, functor);
}

static void LGR_NOINLINE zero_v(
    counting_range<node_index> const /*nodes*/,
    hpc::device_array_vector<vector3<double>, node_index> const& /*x_vector*/,
    hpc::device_array_vector<vector3<double>, node_index>* v) {
  lgr::fill(*v, vector3<double>::zero());
}

static void LGR_NOINLINE spin_v(
    counting_range<node_index> const nodes,
    hpc::device_array_vector<vector3<double>, node_index> const& x_vector,
    hpc::device_array_vector<vector3<double>, node_index>* v_vector) {
  auto const nodes_to_x = x_vector.cbegin();
  auto const nodes_to_v = v_vector->begin();
  auto functor = [=](node_index const node) {
    auto const x = nodes_to_x[node].load();
    nodes_to_v[node] = 100.0 * vector3<double>(-(x(1) - 0.5), (x(0) - 0.5), 0.0);
  };
  lgr::for_each(nodes, functor);
}

static void LGR_NOINLINE elastic_wave() {
  constexpr material_index body(0);
  constexpr material_index nmaterials(1);
  constexpr material_index x_boundary(1);
  constexpr material_index nboundaries(1);
  input in(nmaterials, nboundaries);
  in.name = "elastic_wave";
  in.element = BAR;
  in.end_time = 4.0e-3;
  in.num_file_outputs = 200;
  in.elements_along_x = 1000;
  in.rho0[body] = 1000.0;
  in.enable_neo_Hookean[body] = true;
  in.K0[body] = 1.0e9;
  in.G0[body] = 0.0;
  in.initial_v = set_exponential_wave_v;
  constexpr auto x_axis = vector3<double>::x_axis();
  static constexpr double eps = 1.0e-10;
  auto x_domain = std::make_unique<union_domain>();
  x_domain->add(epsilon_around_plane_domain({x_axis, 0.0}, eps));
  x_domain->add(epsilon_around_plane_domain({x_axis, in.x_domain_size}, eps));
  in.domains[x_boundary] = std::move(x_domain);
  in.zero_acceleration_conditions.push_back({x_boundary, x_axis});
//in.enable_nodal_pressure = true;
//in.c_tau = 0.5;
  run(in);
}

static void LGR_NOINLINE gas_expansion() {
  constexpr material_index gas(0);
  constexpr material_index nmaterials(1);
  constexpr material_index nboundaries(0);
  input in(nmaterials, nboundaries);
  in.name = "gas_expansion";
  in.element = BAR;
  in.end_time = 10.0;
  in.num_file_outputs = 100;
  in.elements_along_x = 160;
  in.rho0[gas] = 1.0;
  in.enable_ideal_gas[gas] = true;
  in.gamma[gas] = 1.4;
  in.e0[gas] = 1.0;
  in.initial_v = zero_v;
  run(in);
}

static void LGR_NOINLINE spinning_square() {
  constexpr material_index body(0);
  constexpr material_index nmaterials(1);
  constexpr material_index nboundaries(0);
  input in(nmaterials, nboundaries);
  in.name = "spinning_square";
  in.element = TRIANGLE;
  in.end_time = 1.0e-2;
  in.num_file_outputs = 400;
  in.elements_along_x = 1;
  in.x_domain_size = 1.0;
  in.elements_along_y = 1;
  in.y_domain_size = 1.0;
  in.rho0[body] = 1000.0;
  in.enable_neo_Hookean[body] = true;
  in.K0[body] = 200.0e9;
  in.G0[body] = 75.0e9;
  in.initial_v = spin_v;
  run(in);
}

static void LGR_NOINLINE quadratic_in_x_v(
    counting_range<node_index> const nodes,
    hpc::device_array_vector<vector3<double>, node_index> const& x_vector,
    hpc::device_array_vector<vector3<double>, node_index>* v_vector) {
  auto const nodes_to_x = x_vector.cbegin();
  auto const nodes_to_v = v_vector->begin();
  auto functor = [=](node_index const node) {
    auto const x = nodes_to_x[node].load();
    auto const norm_x = x(0) / 48.0;
    auto const v_y = norm_x * norm_x;
    nodes_to_v[node] = vector3<double>(0.0, v_y, 0.0);
  };
  lgr::for_each(nodes, functor);
}

static void LGR_NOINLINE Cooks_membrane_x(
    hpc::device_array_vector<vector3<double>, node_index>* x_vector) {
  counting_range<node_index> const nodes(x_vector->size());
  auto const nodes_to_x = x_vector->begin();
  auto functor = [=](node_index const node) {
    auto const unit_x = nodes_to_x[node].load();
    vector3<double> const new_x(
        unit_x(0) * 48.0,
        unit_x(1) * 44.0 +
        unit_x(0) * 16.0 +
        unit_x(0) * (1.0 - unit_x(1)) * 28.0,
        0.0);
    nodes_to_x[node] = new_x;
  };
  lgr::for_each(nodes, functor);
}

static void LGR_NOINLINE Cooks_membrane() {
  constexpr material_index body(0);
  constexpr material_index nmaterials(1);
  constexpr material_index x_min(1);
  constexpr material_index nboundaries(1);
  input in(nmaterials, nboundaries);
  in.name = "Cooks_membrane";
  in.element = TRIANGLE;
  in.end_time = 40.0;
  in.num_file_outputs = 200;
  in.elements_along_x = 8;
  in.x_domain_size = 1.0;
  in.elements_along_y = 8;
  in.y_domain_size = 1.0;
  in.rho0[body] = 1.0;
  in.enable_neo_Hookean[body] = true;
  in.K0[body] = 833333.0;
  in.G0[body] = 83.0;
  in.initial_v = quadratic_in_x_v;
  static constexpr vector3<double> x_axis(1.0, 0.0, 0.0);
  static constexpr double eps = 1.0e-10;
  auto x_min_domain = epsilon_around_plane_domain({x_axis, 0.0}, eps);
  in.domains[x_min] = std::move(x_min_domain);
  static constexpr vector3<double> y_axis(0.0, 1.0, 0.0);
  in.zero_acceleration_conditions.push_back({x_min, x_axis});
  in.zero_acceleration_conditions.push_back({x_min, y_axis});
  in.x_transform = Cooks_membrane_x;
  in.enable_nodal_pressure[body] = true;
  in.c_tau[body] = 0.5;
  run(in);
}

static void LGR_NOINLINE swinging_plate() {
  constexpr material_index body(0);
  constexpr material_index nmaterials(1);
  constexpr material_index x_min(1);
  constexpr material_index x_max(2);
  constexpr material_index y_min(3);
  constexpr material_index y_max(4);
  constexpr material_index nboundaries(4);
  input in(nmaterials, nboundaries);
  in.name = "swinging_plate";
  in.element = TRIANGLE;
  in.num_file_outputs = 200;
  in.elements_along_x = 8;
  in.x_domain_size = 2.0;
  in.elements_along_y = 8;
  in.y_domain_size = 2.0;
  double const rho = 1.1e3;
  in.rho0[body] = rho;
  in.enable_neo_Hookean[body] = true;
  double const nu = 0.45;
  double const E = 1.7e7;
  double const K = E / (3.0 * (1.0 - 2.0 * nu));
  double const G = E / (2.0 * (1.0 + nu));
  double const w = (pi / 2.0) * std::sqrt((2.0 * G) / rho);
  in.end_time = 0.16;
  in.K0[body] = K;
  in.G0[body] = G;
  auto swinging_plate_v = [=] (
    counting_range<node_index> const nodes,
    hpc::device_array_vector<vector3<double>, node_index> const& x_vector,
    hpc::device_array_vector<vector3<double>, node_index>* v_vector) {
    auto const nodes_to_x = x_vector.cbegin();
    auto const nodes_to_v = v_vector->begin();
    double const U0 = 0.001;
    auto functor = [=](node_index const node) {
      auto const x = nodes_to_x[node].load();
      auto const v = (U0 * w) * vector3<double>(
          -std::sin((pi * x(0)) / 2.0) * std::cos((pi * x(1)) / 2.0),
          std::cos((pi * x(0)) / 2.0) * std::sin((pi * x(1)) / 2.0),
          0.0);
      nodes_to_v[node] = v;
    };
    lgr::for_each(nodes, functor);
  };
  in.initial_v = swinging_plate_v;
  static constexpr vector3<double> x_axis(1.0, 0.0, 0.0);
  static constexpr vector3<double> y_axis(0.0, 1.0, 0.0);
  static constexpr double eps = 1.0e-10;
  in.domains[x_min] = epsilon_around_plane_domain({x_axis, 0.0}, eps);
  in.domains[x_max] = epsilon_around_plane_domain({x_axis, in.x_domain_size}, eps);
  in.domains[y_min] = epsilon_around_plane_domain({y_axis, 0.0}, eps);
  in.domains[y_max] = epsilon_around_plane_domain({y_axis, in.y_domain_size}, eps);
  in.zero_acceleration_conditions.push_back({x_min, x_axis});
  in.zero_acceleration_conditions.push_back({y_min, y_axis});
  in.zero_acceleration_conditions.push_back({x_max, x_axis});
  in.zero_acceleration_conditions.push_back({y_max, y_axis});
  in.enable_nodal_pressure[body] = true;
  in.c_tau[body] = 0.5;
  run(in);
}

static void LGR_NOINLINE spinning_cube() {
  constexpr material_index body(0);
  constexpr material_index nmaterials(1);
  constexpr material_index nboundaries(0);
  input in(nmaterials, nboundaries);
  in.name = "spinning_cube";
  in.element = TETRAHEDRON;
  in.end_time = 1.0e-2;
  in.num_file_outputs = 400;
  in.elements_along_x = 1;
  in.x_domain_size = 1.0;
  in.elements_along_y = 1;
  in.y_domain_size = 1.0;
  in.elements_along_z = 1;
  in.z_domain_size = 1.0;
  in.rho0[body] = 7800.0;
  in.enable_neo_Hookean[body] = true;
  in.K0[body] = 200.0e9;
  in.G0[body] = 75.0e9;
  in.initial_v = spin_v;
  in.CFL = 0.9;
  in.time_integrator = VELOCITY_VERLET;
  run(in);
}

static void LGR_NOINLINE elastic_wave_2d() {
  constexpr material_index body(0);
  constexpr material_index nmaterials(1);
  constexpr material_index x_boundary(1);
  constexpr material_index y_boundary(2);
  constexpr material_index nboundaries(2);
  input in(nmaterials, nboundaries);
  in.name = "elastic_wave_2d";
  in.element = TRIANGLE;
  in.end_time = 2.0e-3;
  in.num_file_outputs = 100;
  in.elements_along_x = 1000;
  in.elements_along_y = 1;
  in.y_domain_size = 1.0e-3;
  in.rho0[body] = 1000.0;
  in.enable_neo_Hookean[body] = true;
  in.K0[body] = 1.0e9;
  in.G0[body] = 0.0;
  in.initial_v = set_exponential_wave_v;
  static constexpr vector3<double> x_axis(1.0, 0.0, 0.0);
  static constexpr vector3<double> y_axis(0.0, 1.0, 0.0);
  static constexpr double eps = 1.0e-10;
  auto x_domain = std::make_unique<union_domain>();
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

static void LGR_NOINLINE elastic_wave_3d() {
  constexpr material_index body(0);
  constexpr material_index nmaterials(1);
  constexpr material_index x_boundary(1);
  constexpr material_index y_boundary(2);
  constexpr material_index z_boundary(3);
  constexpr material_index nboundaries(3);
  input in(nmaterials, nboundaries);
  in.name = "elastic_wave_3d";
  in.element = TETRAHEDRON;
  in.end_time = 2.0e-3;
  in.num_file_outputs = 100;
  in.elements_along_x = 1000;
  in.elements_along_y = 1;
  in.y_domain_size = 1.0e-3;
  in.elements_along_z = 1;
  in.z_domain_size = 1.0e-3;
  in.rho0[body] = 1000.0;
  in.enable_neo_Hookean[body] = true;
  in.K0[body] = 1.0e9;
  in.G0[body] = 0.0;
  in.initial_v = set_exponential_wave_v;
  static constexpr vector3<double> x_axis(1.0, 0.0, 0.0);
  static constexpr vector3<double> y_axis(0.0, 1.0, 0.0);
  static constexpr vector3<double> z_axis(0.0, 0.0, 1.0);
  static constexpr double eps = 1.0e-10;
  auto x_domain = std::make_unique<union_domain>();
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

static void LGR_NOINLINE swinging_cube(bool stabilize) {
  constexpr material_index body(0);
  constexpr material_index nmaterials(1);
  constexpr material_index x_min(1);
  constexpr material_index x_max(2);
  constexpr material_index y_min(3);
  constexpr material_index y_max(4);
  constexpr material_index z_min(5);
  constexpr material_index z_max(6);
  constexpr material_index nboundaries(6);
  input in(nmaterials, nboundaries);
  if (stabilize) {
  in.name = "stabilized_swinging_cube";
  } else {
  in.name = "swinging_cube";
  }
  in.element = TETRAHEDRON;
  in.num_file_outputs = 100;
  in.elements_along_x = 8;
  in.x_domain_size = 2.0;
  in.elements_along_y = 8;
  in.y_domain_size = 2.0;
  in.elements_along_z = 8;
  in.z_domain_size = 2.0;
  double const rho = 1.1e3;
  in.rho0[body] = rho;
  in.enable_neo_Hookean[body] = true;
  double const nu = 0.45;
  double const E = 1.7e7;
  double const K = E / (3.0 * (1.0 - 2.0 * nu));
  double const G = E / (2.0 * (1.0 + nu));
  double const w = pi * std::sqrt((3.0 * G) / (4.0 * rho));
  in.end_time = 0.10;
  in.K0[body] = K;
  in.G0[body] = G;
  auto swinging_cube_v = [=] (
    counting_range<node_index> const nodes,
    hpc::device_array_vector<vector3<double>, node_index> const& x_vector,
    hpc::device_array_vector<vector3<double>, node_index>* v_vector) {
    auto const nodes_to_x = x_vector.cbegin();
    auto const nodes_to_v = v_vector->begin();
    double const U0 = 5.0e-4;
    auto functor = [=](node_index const node) {
      constexpr double half_pi = pi / 2.0;
      auto const x = nodes_to_x[node].load();
      auto const v = (U0 * w) * vector3<double>(
          -std::sin(half_pi * x(0)) * std::cos(half_pi * x(1)) * std::cos(half_pi * x(2)),
          std::cos(half_pi * x(0)) * std::sin(half_pi * x(1)) * std::cos(half_pi * x(2)),
          std::cos(half_pi * x(0)) * std::cos(half_pi * x(1)) * std::sin(half_pi * x(2)));
      nodes_to_v[node] = v;
    };
    lgr::for_each(nodes, functor);
  };
  in.initial_v = swinging_cube_v;
  static constexpr vector3<double> x_axis(1.0, 0.0, 0.0);
  static constexpr vector3<double> y_axis(0.0, 1.0, 0.0);
  static constexpr vector3<double> z_axis(0.0, 0.0, 1.0);
  static constexpr double eps = 1.0e-10;
  in.domains[x_min] = epsilon_around_plane_domain({x_axis, 0.0}, eps);
  in.domains[x_max] = epsilon_around_plane_domain({x_axis, in.x_domain_size}, eps);
  in.domains[y_min] = epsilon_around_plane_domain({y_axis, 0.0}, eps);
  in.domains[y_max] = epsilon_around_plane_domain({y_axis, in.y_domain_size}, eps);
  in.domains[z_min] = epsilon_around_plane_domain({z_axis, 0.0}, eps);
  in.domains[z_max] = epsilon_around_plane_domain({z_axis, in.z_domain_size}, eps);
  in.zero_acceleration_conditions.push_back({x_min, x_axis});
  in.zero_acceleration_conditions.push_back({x_max, x_axis});
  in.zero_acceleration_conditions.push_back({y_min, y_axis});
  in.zero_acceleration_conditions.push_back({y_max, y_axis});
  in.zero_acceleration_conditions.push_back({z_min, z_axis});
  in.zero_acceleration_conditions.push_back({z_max, z_axis});
  in.enable_nodal_pressure[body] = stabilize;
  in.c_tau[body] = 0.5;
  in.CFL = 0.45;
  run(in);
}

static void LGR_NOINLINE twisting_column() {
  constexpr material_index body(0);
  constexpr material_index nmaterials(1);
  constexpr material_index y_min(1);
  constexpr material_index nboundaries(1);
  input in(nmaterials, nboundaries);
  in.name = "twisting_column";
  in.element = TETRAHEDRON;
  in.end_time = 0.1;
  in.num_file_outputs = 100;
  in.elements_along_x = 3;
  in.x_domain_size = 1.0;
  in.elements_along_y = 18;
  in.y_domain_size = 6.0;
  in.elements_along_z = 3;
  in.z_domain_size = 1.0;
  double const rho = 1.1e3;
  in.rho0[body] = rho;
  in.enable_neo_Hookean[body] = true;
  double const nu = 0.499;
  double const E = 1.7e7;
  double const K = E / (3.0 * (1.0 - 2.0 * nu));
  double const G = E / (2.0 * (1.0 + nu));
  in.K0[body] = K;
  in.G0[body] = G;
  auto twisting_column_v = [=] (
    counting_range<node_index> const nodes,
    hpc::device_array_vector<vector3<double>, node_index> const& x_vector,
    hpc::device_array_vector<vector3<double>, node_index>* v_vector) {
    auto const nodes_to_x = x_vector.cbegin();
    auto const nodes_to_v = v_vector->begin();
    auto functor = [=](node_index const node) {
      auto const x = nodes_to_x[node].load();
      auto const v = 100.0 * std::sin((pi / 12.0) * x(1)) * vector3<double>((x(2) - 0.5), 0.0, -(x(0) - 0.5));
      nodes_to_v[node] = v;
    };
    lgr::for_each(nodes, functor);
  };
  in.initial_v = twisting_column_v;
  static constexpr vector3<double> x_axis(1.0, 0.0, 0.0);
  static constexpr vector3<double> y_axis(0.0, 1.0, 0.0);
  static constexpr vector3<double> z_axis(0.0, 0.0, 1.0);
  static constexpr double eps = 1.0e-10;
  in.domains[y_min] = epsilon_around_plane_domain({y_axis, 0.0}, eps);
  in.zero_acceleration_conditions.push_back({y_min, x_axis});
  in.zero_acceleration_conditions.push_back({y_min, y_axis});
  in.zero_acceleration_conditions.push_back({y_min, z_axis});
  in.enable_nodal_pressure[body] = true;
  in.c_tau[body] = 0.5;
  in.CFL = 0.9;
  run(in);
}

static void LGR_NOINLINE Noh_1D() {
  constexpr material_index gas(0);
  constexpr material_index nmaterials(1);
  constexpr material_index x_min(1);
  constexpr material_index nboundaries(1);
  input in(nmaterials, nboundaries);
  in.name = "Noh_1D";
  in.element = BAR;
  in.end_time = 0.6;
  in.num_file_outputs = 60;
  in.elements_along_x = 44;
  in.x_domain_size = 1.1;
  in.rho0[gas] = 1.0;
  in.enable_ideal_gas[gas] = true;
  in.gamma[gas] = 5.0 / 3.0;
  in.e0[gas] = 1.0e-4;
  auto inward_v = [=] (
    counting_range<node_index> const nodes,
    hpc::device_array_vector<vector3<double>, node_index> const& x_vector,
    hpc::device_array_vector<vector3<double>, node_index>* v_vector) {
    auto const nodes_to_x = x_vector.cbegin();
    auto const nodes_to_v = v_vector->begin();
    auto functor = [=](node_index const node) {
      auto const x = nodes_to_x[node].load();
      auto const n = norm(x);
      auto const v = (n == 0) ? vector3<double>::zero() : (-(x / n));
      nodes_to_v[node] = v;
    };
    lgr::for_each(nodes, functor);
  };
  in.initial_v = inward_v;
  static constexpr vector3<double> x_axis(1.0, 0.0, 0.0);
  static constexpr double eps = 1.0e-10;
  in.domains[x_min] = epsilon_around_plane_domain({x_axis, 0.0}, eps);
  in.zero_acceleration_conditions.push_back({x_min, x_axis});
  in.enable_viscosity = true;
  in.linear_artificial_viscosity = 1.0;
  in.quadratic_artificial_viscosity = 1.0;
  in.enable_nodal_energy[gas] = false;
  in.c_tau[gas] = 0.0;
  in.CFL = 0.9;
  run(in);
}

static void LGR_NOINLINE Noh_2D() {
  constexpr material_index gas(0);
  constexpr material_index nmaterials(1);
  constexpr material_index x_min(1);
  constexpr material_index y_min(2);
  constexpr material_index nboundaries(2);
  input in(nmaterials, nboundaries);
  in.name = "Noh_2D";
  in.element = TRIANGLE;
  in.end_time = 0.6;
  in.num_file_outputs = 60;
  in.elements_along_x = 34;
  in.x_domain_size = 0.85;
  in.elements_along_y = 34;
  in.y_domain_size = 0.85;
  in.rho0[gas] = 1.0;
  in.enable_ideal_gas[gas] = true;
  in.gamma[gas] = 5.0 / 3.0;
  in.e0[gas] = 1.0e-14;
  auto inward_v = [=] (
    counting_range<node_index> const nodes,
    hpc::device_array_vector<vector3<double>, node_index> const& x_vector,
    hpc::device_array_vector<vector3<double>, node_index>* v_vector) {
    auto const nodes_to_x = x_vector.cbegin();
    auto const nodes_to_v = v_vector->begin();
    auto functor = [=](node_index const node) {
      auto const x = nodes_to_x[node].load();
      auto const n = norm(x);
      auto const v = (n == 0) ? vector3<double>::zero() : (-(x / n));
      nodes_to_v[node] = v;
    };
    lgr::for_each(nodes, functor);
  };
  in.initial_v = inward_v;
  static constexpr vector3<double> x_axis(1.0, 0.0, 0.0);
  static constexpr vector3<double> y_axis(0.0, 1.0, 0.0);
  static constexpr double eps = 1.0e-10;
  in.domains[x_min] = epsilon_around_plane_domain({x_axis, 0.0}, eps);
  in.domains[y_min] = epsilon_around_plane_domain({y_axis, 0.0}, eps);
  in.zero_acceleration_conditions.push_back({x_min, x_axis});
  in.zero_acceleration_conditions.push_back({y_min, y_axis});
  in.enable_viscosity = true;
  in.linear_artificial_viscosity = 1.0;
  in.quadratic_artificial_viscosity = 0.5;
  in.enable_nodal_energy[gas] = true;
  in.c_tau[gas] = 1.0;
  run(in);
}

static void LGR_NOINLINE spinning_composite_cube() {
  constexpr material_index body(0);
  constexpr material_index nmaterials(1);
  constexpr material_index nboundaries(0);
  input in(nmaterials, nboundaries);
  in.name = "spinning_composite_cube";
  in.element = COMPOSITE_TETRAHEDRON;
  in.end_time = 1.0e-2;
  in.num_file_outputs = 400;
  in.elements_along_x = 1;
  in.x_domain_size = 1.0;
  in.elements_along_y = 1;
  in.y_domain_size = 1.0;
  in.elements_along_z = 1;
  in.z_domain_size = 1.0;
  in.rho0[body] = 7800.0;
  in.enable_neo_Hookean[body] = true;
  in.K0[body] = 200.0e9;
  in.G0[body] = 75.0e9;
  in.initial_v = spin_v;
  in.CFL = 0.9;
  in.time_integrator = VELOCITY_VERLET;
  run(in);
}

static void LGR_NOINLINE twisting_composite_column() {
  constexpr material_index body(0);
  constexpr material_index nmaterials(1);
  constexpr material_index y_min(1);
  constexpr material_index nboundaries(1);
  input in(nmaterials, nboundaries);
  in.name = "twisting_composite_column";
  in.element = COMPOSITE_TETRAHEDRON;
  in.end_time = 0.1;
  in.num_file_outputs = 100;
  in.elements_along_x = 3;
  in.x_domain_size = 1.0;
  in.elements_along_y = 18;
  in.y_domain_size = 6.0;
  in.elements_along_z = 3;
  in.z_domain_size = 1.0;
  double const rho = 1.1e3;
  in.rho0[body] = rho;
  in.enable_neo_Hookean[body] = true;
  double const nu = 0.499;
  double const E = 1.7e7;
  double const K = E / (3.0 * (1.0 - 2.0 * nu));
  double const G = E / (2.0 * (1.0 + nu));
  in.K0[body] = K;
  in.G0[body] = G;
  auto twisting_column_v = [=] (
    counting_range<node_index> const nodes,
    hpc::device_array_vector<vector3<double>, node_index> const& x_vector,
    hpc::device_array_vector<vector3<double>, node_index>* v_vector) {
    auto const nodes_to_x = x_vector.cbegin();
    auto const nodes_to_v = v_vector->begin();
    auto functor = [=](node_index const node) {
      auto const x = nodes_to_x[node].load();
      auto const v = 100.0 * std::sin((pi / 12.0) * x(1)) * vector3<double>((x(2) - 0.5), 0.0, -(x(0) - 0.5));
      nodes_to_v[node] = v;
    };
    lgr::for_each(nodes, functor);
  };
  in.initial_v = twisting_column_v;
  static constexpr vector3<double> x_axis(1.0, 0.0, 0.0);
  static constexpr vector3<double> y_axis(0.0, 1.0, 0.0);
  static constexpr vector3<double> z_axis(0.0, 0.0, 1.0);
  static constexpr double eps = 1.0e-10;
  in.domains[y_min] = epsilon_around_plane_domain({y_axis, 0.0}, eps);
  in.zero_acceleration_conditions.push_back({y_min, x_axis});
  in.zero_acceleration_conditions.push_back({y_min, y_axis});
  in.zero_acceleration_conditions.push_back({y_min, z_axis});
  in.enable_J_averaging = true;
  in.CFL = 0.9;
  run(in);
}

static void LGR_NOINLINE Noh_3D() {
  constexpr material_index gas(0);
  constexpr material_index nmaterials(1);
  constexpr material_index x_min(1);
  constexpr material_index y_min(2);
  constexpr material_index z_min(3);
  constexpr material_index nboundaries(3);
  input in(nmaterials, nboundaries);
  in.name = "Noh_3D";
  in.element = TETRAHEDRON;
  in.end_time = 0.6;
  in.num_file_outputs = 10;
  in.elements_along_x = 20;
  in.x_domain_size = 0.9;
  in.elements_along_y = 20;
  in.y_domain_size = 0.9;
  in.elements_along_z = 20;
  in.z_domain_size = 0.9;
  in.rho0[gas] = 1.0;
  in.enable_ideal_gas[gas] = true;
  in.gamma[gas] = 5.0 / 3.0;
  in.e0[gas] = 1.0e-14;
  auto inward_v = [=] (
    counting_range<node_index> const nodes,
    hpc::device_array_vector<vector3<double>, node_index> const& x_vector,
    hpc::device_array_vector<vector3<double>, node_index>* v_vector) {
    auto const nodes_to_x = x_vector.cbegin();
    auto const nodes_to_v = v_vector->begin();
    auto functor = [=](node_index const node) {
      auto const x = nodes_to_x[node].load();
      auto const n = norm(x);
      auto const v = (n == 0) ? vector3<double>::zero() : (-(x / n));
      nodes_to_v[node] = v;
    };
    lgr::for_each(nodes, functor);
  };
  in.initial_v = inward_v;
  static constexpr vector3<double> x_axis(1.0, 0.0, 0.0);
  static constexpr vector3<double> y_axis(0.0, 1.0, 0.0);
  static constexpr vector3<double> z_axis(0.0, 0.0, 1.0);
  static constexpr double eps = 1.0e-10;
  in.domains[x_min] = epsilon_around_plane_domain({x_axis, 0.0}, eps);
  in.domains[y_min] = epsilon_around_plane_domain({y_axis, 0.0}, eps);
  in.domains[z_min] = epsilon_around_plane_domain({z_axis, 0.0}, eps);
  in.zero_acceleration_conditions.push_back({x_min, x_axis});
  in.zero_acceleration_conditions.push_back({y_min, y_axis});
  in.zero_acceleration_conditions.push_back({z_min, z_axis});
  in.enable_viscosity = true;
  in.linear_artificial_viscosity = 1.0;
  in.quadratic_artificial_viscosity = 0.1;
  in.enable_nodal_energy[gas] = true;
  in.c_tau[gas] = 1.0;
  run(in);
}

static void LGR_NOINLINE composite_Noh_3D() {
  constexpr material_index gas(0);
  constexpr material_index nmaterials(1);
  constexpr material_index x_min(1);
  constexpr material_index y_min(2);
  constexpr material_index z_min(3);
  constexpr material_index nboundaries(3);
  input in(nmaterials, nboundaries);
  in.name = "composite_Noh_3D";
  in.element = COMPOSITE_TETRAHEDRON;
  in.end_time = 0.6;
  in.num_file_outputs = 10;
  in.elements_along_x = 20;
  in.x_domain_size = 0.9;
  in.elements_along_y = 20;
  in.y_domain_size = 0.9;
  in.elements_along_z = 20;
  in.z_domain_size = 0.9;
  in.rho0[gas] = 1.0;
  in.enable_ideal_gas[gas] = true;
  in.gamma[gas] = 5.0 / 3.0;
  in.e0[gas] = 1.0e-14;
  auto inward_v = [=] (
    counting_range<node_index> const nodes,
    hpc::device_array_vector<vector3<double>, node_index> const& x_vector,
    hpc::device_array_vector<vector3<double>, node_index>* v_vector) {
    auto const nodes_to_x = x_vector.cbegin();
    auto const nodes_to_v = v_vector->begin();
    auto functor = [=](node_index const node) {
      auto const x = nodes_to_x[node].load();
      auto const n = norm(x);
      auto const v = (n == 0) ? vector3<double>::zero() : (-(x / n));
      nodes_to_v[node] = v;
    };
    lgr::for_each(nodes, functor);
  };
  in.initial_v = inward_v;
  static constexpr vector3<double> x_axis(1.0, 0.0, 0.0);
  static constexpr vector3<double> y_axis(0.0, 1.0, 0.0);
  static constexpr vector3<double> z_axis(0.0, 0.0, 1.0);
  static constexpr double eps = 1.0e-10;
  in.domains[x_min] = epsilon_around_plane_domain({x_axis, 0.0}, eps);
  in.domains[y_min] = epsilon_around_plane_domain({y_axis, 0.0}, eps);
  in.domains[z_min] = epsilon_around_plane_domain({z_axis, 0.0}, eps);
  in.zero_acceleration_conditions.push_back({x_min, x_axis});
  in.zero_acceleration_conditions.push_back({y_min, y_axis});
  in.zero_acceleration_conditions.push_back({z_min, z_axis});
  in.enable_viscosity = true;
  in.linear_artificial_viscosity = 0.25;
  in.quadratic_artificial_viscosity = 0.5;
  in.enable_p_averaging = false;
  in.enable_rho_averaging = false;
  in.enable_e_averaging = true;
  run(in);
}

static void LGR_NOINLINE Sod_1D() {
  constexpr material_index left(0);
  constexpr material_index right(1);
  constexpr material_index nmaterials(2);
  constexpr material_index x_min(2);
  constexpr material_index x_max(3);
  constexpr material_index nboundaries(2);
  input in(nmaterials, nboundaries);
  in.name = "Sod_1D";
  in.element = BAR;
  in.end_time = 0.14;
  in.num_file_outputs = 14;
  in.elements_along_x = 100;
  in.x_domain_size = 1.0;
  in.rho0[left] = 1.0;
  in.rho0[right] = 0.125;
  in.enable_ideal_gas[left] = true;
  in.enable_ideal_gas[right] = true;
  in.gamma[left] = 1.4;
  in.gamma[right] = 1.4;
  in.e0[left] = 1.0 / ((1.4 - 1.0) * 1.0);
  in.e0[right] = 0.1 / ((1.4 - 1.0) * 0.125);
  in.initial_v = zero_v;
  static constexpr vector3<double> x_axis(1.0, 0.0, 0.0);
  static constexpr double eps = 1.0e-10;
  in.domains[x_min] = epsilon_around_plane_domain({x_axis, 0.0}, eps);
  in.domains[x_max] = epsilon_around_plane_domain({x_axis, 0.0}, eps);
  in.zero_acceleration_conditions.push_back({x_min, x_axis});
  in.zero_acceleration_conditions.push_back({x_max, x_axis});
  in.enable_viscosity = true;
  in.linear_artificial_viscosity = 0.5;
  in.quadratic_artificial_viscosity = 0.125;
  auto right_domain = half_space_domain(plane{vector3<double>{1.0, 0.0, 0.0}, 0.5});
  auto left_domain = half_space_domain(plane{vector3<double>{-1.0, 0.0, 0.0}, -0.5});
  in.domains[left] = std::move(left_domain);
  in.domains[right] = std::move(right_domain);
  in.enable_nodal_energy[left] = true;
  in.enable_nodal_energy[right] = true;
  in.c_tau[left] = 0.0;
  in.c_tau[right] = 0.0;
  run(in);
}

static void LGR_NOINLINE triple_point() {
  constexpr material_index left(0);
  constexpr material_index right_bottom(1);
  constexpr material_index right_top(2);
  constexpr material_index nmaterials(3);
  constexpr material_index x_min(3);
  constexpr material_index x_max(4);
  constexpr material_index y_min(5);
  constexpr material_index y_max(6);
  constexpr material_index nboundaries(4);
  input in(nmaterials, nboundaries);
  in.name = "triple_point";
  in.element = TRIANGLE;
  in.end_time = 6.0;
  in.num_file_outputs = 60;
  in.elements_along_x = 112;
  in.x_domain_size = 7.0;
  in.elements_along_y = 48;
  in.y_domain_size = 3.0;
  in.rho0[right_top] = 0.1;
  in.rho0[right_bottom] = 1.0;
  in.rho0[left] = 1.0;
  in.enable_ideal_gas[left] = true;
  in.enable_ideal_gas[right_top] = true;
  in.enable_ideal_gas[right_bottom] = true;
  in.gamma[right_top] = 1.5;
  in.gamma[left] = 1.5;
  in.gamma[right_bottom] = 1.4;
  in.e0[right_top] = 2.5;
  in.e0[right_bottom] = 0.3125;
  in.e0[left] = 2.0;
  in.initial_v = zero_v;
  constexpr auto x_axis = vector3<double>::x_axis();
  constexpr auto y_axis = vector3<double>::y_axis();
  constexpr double eps = 1.0e-10;
  in.domains[x_min] = epsilon_around_plane_domain({x_axis, 0.0}, eps);
  in.domains[x_max] = epsilon_around_plane_domain({x_axis, in.x_domain_size}, eps);
  in.domains[y_min] = epsilon_around_plane_domain({y_axis, 0.0}, eps);
  in.domains[y_max] = epsilon_around_plane_domain({y_axis, in.y_domain_size}, eps);
  in.zero_acceleration_conditions.push_back({x_min, x_axis});
  in.zero_acceleration_conditions.push_back({x_max, x_axis});
  in.zero_acceleration_conditions.push_back({y_min, y_axis});
  in.zero_acceleration_conditions.push_back({y_max, y_axis});
  auto left_domain = box_domain({0.0, 0.0, -eps}, {1.0, 3.0, eps});
  auto right_bottom_domain = box_domain({1.0, 0.0, -eps}, {7.0, 1.5, eps});
  auto right_top_domain = box_domain({1.0, 1.5, -eps}, {7.0, 3.0, eps});
  in.domains[left] = std::move(left_domain);
  in.domains[right_bottom] = std::move(right_bottom_domain);
  in.domains[right_top] = std::move(right_top_domain);
  in.enable_viscosity = true;
  in.linear_artificial_viscosity = 0.5;
  in.enable_nodal_energy[left] = true;
  in.enable_nodal_energy[right_bottom] = true;
  in.enable_nodal_energy[right_top] = true;
  in.c_tau[left] = 1.0;
  in.c_tau[right_bottom] = 1.0;
  in.c_tau[right_top] = 1.0;
  in.enable_adapt = true;
  run(in);
}

}

int main() {
  if ((0)) lgr::elastic_wave();
  if ((0)) lgr::gas_expansion();
  if ((0)) lgr::spinning_square();
  if ((0)) lgr::Cooks_membrane();
  if ((0)) lgr::swinging_plate();
  if ((0)) lgr::spinning_cube();
  if ((0)) lgr::elastic_wave_2d();
  if ((0)) lgr::elastic_wave_3d();
  if ((0)) lgr::swinging_cube(true);
  if ((0)) lgr::swinging_cube(false);
  if ((0)) lgr::twisting_column();
  if ((0)) lgr::Noh_1D();
  if ((0)) lgr::Noh_2D();
  if ((0)) lgr::Noh_3D();
  if ((0)) lgr::composite_Noh_3D();
  if ((0)) lgr::spinning_composite_cube();
  if ((1)) lgr::twisting_composite_column();
  if ((0)) lgr::Sod_1D();
  if ((0)) lgr::triple_point();
}
