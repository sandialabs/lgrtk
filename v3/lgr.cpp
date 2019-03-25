#include <memory>

#include <lgr_run.hpp>
#include <lgr_for_each.hpp>
#include <lgr_fill.hpp>
#include <lgr_domain.hpp>
#include <lgr_input.hpp>

// DEBUG
#include <iostream>

namespace lgr {

static std::unique_ptr<domain> epsilon_around_plane_domain(plane const& p, double eps) {
  auto out = std::make_unique<clipped_domain<all_space>>(all_space{});
  out->clip({p.normal, p.origin - eps});
  out->clip({-p.normal, -p.origin - eps});
  return out;
}

static void LGR_NOINLINE set_exponential_wave_v(
    int_range const nodes,
    host_vector<vector3<double>> const& x_vector,
    host_vector<vector3<double>>* v_vector) {
  auto const nodes_to_x = x_vector.cbegin();
  auto const nodes_to_v = v_vector->begin();
  auto functor = [=](int const node) {
    vector3<double> const x = nodes_to_x[node];
    auto const d = x(0) - 0.5;
    auto const v_x = 1.0e-4 * std::exp(-(d * d) / (2 * (0.05 * 0.05)));
    nodes_to_v[node] = vector3<double>(v_x, 0.0, 0.0);
  };
  lgr::for_each(nodes, functor);
}

static void LGR_NOINLINE zero_v(
    int_range const /*nodes*/,
    host_vector<vector3<double>> const& /*x_vector*/,
    host_vector<vector3<double>>* v) {
  lgr::fill(*v, vector3<double>::zero());
}

static void LGR_NOINLINE spin_v(
    int_range const nodes,
    host_vector<vector3<double>> const& x_vector,
    host_vector<vector3<double>>* v_vector) {
  auto const nodes_to_x = x_vector.cbegin();
  auto const nodes_to_v = v_vector->begin();
  auto functor = [=](int const node) {
    vector3<double> const x = nodes_to_x[node];
    nodes_to_v[node] = 100.0 * vector3<double>(-(x(1) - 0.5), (x(0) - 0.5), 0.0);
  };
  lgr::for_each(nodes, functor);
}

static void LGR_NOINLINE run_elastic_wave() {
  input in;
  in.name = "elastic_wave";
  in.element = BAR;
  in.end_time = 4.0e-3;
  in.num_file_outputs = 200;
  in.elements_along_x = 1000;
  in.rho0 = 1000.0;
  in.enable_neo_Hookean = true;
  in.K0 = 1.0e9;
  in.G0 = 0.0;
  in.initial_v = set_exponential_wave_v;
  static constexpr vector3<double> x_axis(1.0, 0.0, 0.0);
  static constexpr double eps = 1.0e-10;
  auto x_minmax = std::make_unique<union_domain>();
  x_minmax->add(epsilon_around_plane_domain({x_axis, 0.0}, eps));
  x_minmax->add(epsilon_around_plane_domain({x_axis, in.x_domain_size}, eps));
  in.node_sets["x_minmax"] = std::move(x_minmax);
  in.zero_acceleration_conditions.push_back({"x_minmax", x_axis});
  run(in);
}

static void LGR_NOINLINE run_gas_expansion() {
  input in;
  in.name = "gas_expansion";
  in.element = BAR;
  in.end_time = 10.0;
  in.num_file_outputs = 100;
  in.elements_along_x = 160;
  in.rho0 = 1.0;
  in.enable_ideal_gas = true;
  in.gamma = 1.4;
  in.e0 = 1.0;
  in.initial_v = zero_v;
  run(in);
}

static void LGR_NOINLINE spinning_square() {
  input in;
  in.name = "spinning_square";
  in.element = TRIANGLE;
  in.end_time = 1.0e-2;
  in.num_file_outputs = 400;
  in.elements_along_x = 1;
  in.x_domain_size = 1.0;
  in.elements_along_y = 1;
  in.y_domain_size = 1.0;
  in.rho0 = 1000.0;
  in.enable_neo_Hookean = true;
  in.K0 = 200.0e9;
  in.G0 = 75.0e9;
  in.initial_v = spin_v;
  run(in);
}

static void LGR_NOINLINE quadratic_in_x_v(
    int_range const nodes,
    host_vector<vector3<double>> const& x_vector,
    host_vector<vector3<double>>* v_vector) {
  auto const nodes_to_x = x_vector.cbegin();
  auto const nodes_to_v = v_vector->begin();
  auto functor = [=](int const node) {
    vector3<double> const x = nodes_to_x[node];
    auto const norm_x = x(0) / 48.0;
    auto const v_y = norm_x * norm_x;
    nodes_to_v[node] = vector3<double>(0.0, v_y, 0.0);
  };
  lgr::for_each(nodes, functor);
}

static void LGR_NOINLINE Cooks_membrane_x(
    host_vector<vector3<double>>* x_vector) {
  int_range const nodes(x_vector->size());
  auto const nodes_to_x = x_vector->begin();
  auto functor = [=](int const node) {
    vector3<double> const unit_x = nodes_to_x[node];
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
  input in;
  in.name = "Cooks_membrane";
  in.element = TRIANGLE;
  in.end_time = 40.0;
  in.num_file_outputs = 200;
  in.elements_along_x = 8;
  in.x_domain_size = 1.0;
  in.elements_along_y = 8;
  in.y_domain_size = 1.0;
  in.rho0 = 1.0;
  in.enable_neo_Hookean = true;
  in.K0 = 833333.0;
  in.G0 = 83.0;
  in.initial_v = quadratic_in_x_v;
  static constexpr vector3<double> x_axis(1.0, 0.0, 0.0);
  static constexpr double eps = 1.0e-10;
  auto x_min = epsilon_around_plane_domain({x_axis, 0.0}, eps);
  in.node_sets["x_min"] = std::move(x_min);
  static constexpr vector3<double> y_axis(0.0, 1.0, 0.0);
  in.zero_acceleration_conditions.push_back({"x_min", x_axis});
  in.zero_acceleration_conditions.push_back({"x_min", y_axis});
  in.x_transform = Cooks_membrane_x;
  in.enable_nodal_pressure = true;
  in.c_tau = 0.5;
  run(in);
}

static void LGR_NOINLINE swinging_plate() {
  input in;
  in.name = "swinging_plate";
  in.element = TRIANGLE;
  in.num_file_outputs = 200;
  in.elements_along_x = 8;
  in.x_domain_size = 2.0;
  in.elements_along_y = 8;
  in.y_domain_size = 2.0;
  double const rho = 1.1e3;
  in.rho0 = rho;
  in.enable_neo_Hookean = true;
  double const nu = 0.45;
  double const E = 1.7e7;
  double const K = E / (3.0 * (1.0 - 2.0 * nu));
  double const G = E / (2.0 * (1.0 + nu));
  double const w = (pi / 2.0) * std::sqrt((2.0 * G) / rho);
  in.end_time = 0.16;
  in.K0 = K;
  in.G0 = G;
  auto swinging_plate_v = [=] (
    int_range const nodes,
    host_vector<vector3<double>> const& x_vector,
    host_vector<vector3<double>>* v_vector) {
    auto const nodes_to_x = x_vector.cbegin();
    auto const nodes_to_v = v_vector->begin();
    double const U0 = 0.001;
    auto functor = [=](int const node) {
      vector3<double> const x = nodes_to_x[node];
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
  in.node_sets["x_min"] = epsilon_around_plane_domain({x_axis, 0.0}, eps);
  in.node_sets["x_max"] = epsilon_around_plane_domain({x_axis, in.x_domain_size}, eps);
  in.node_sets["y_min"] = epsilon_around_plane_domain({y_axis, 0.0}, eps);
  in.node_sets["y_max"] = epsilon_around_plane_domain({y_axis, in.y_domain_size}, eps);
  in.zero_acceleration_conditions.push_back({"x_min", x_axis});
  in.zero_acceleration_conditions.push_back({"y_min", y_axis});
  in.zero_acceleration_conditions.push_back({"x_max", x_axis});
  in.zero_acceleration_conditions.push_back({"y_max", y_axis});
  in.enable_nodal_pressure = true;
  in.c_tau = 0.5;
  run(in);
}

static void LGR_NOINLINE spinning_cube() {
  input in;
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
  in.rho0 = 7800.0;
  in.enable_neo_Hookean = true;
  in.K0 = 200.0e9;
  in.G0 = 75.0e9;
  in.initial_v = spin_v;
  in.CFL = 0.9;
  in.time_integrator = VELOCITY_VERLET;
  run(in);
}

static void LGR_NOINLINE run_elastic_wave_2d() {
  input in;
  in.name = "elastic_wave_2d";
  in.element = TRIANGLE;
  in.end_time = 2.0e-3;
  in.num_file_outputs = 100;
  in.elements_along_x = 1000;
  in.elements_along_y = 1;
  in.y_domain_size = 1.0e-3;
  in.rho0 = 1000.0;
  in.enable_neo_Hookean = true;
  in.K0 = 1.0e9;
  in.G0 = 0.0;
  in.initial_v = set_exponential_wave_v;
  static constexpr vector3<double> x_axis(1.0, 0.0, 0.0);
  static constexpr vector3<double> y_axis(0.0, 1.0, 0.0);
  static constexpr double eps = 1.0e-10;
  auto x_minmax = std::make_unique<union_domain>();
  x_minmax->add(epsilon_around_plane_domain({x_axis, 0.0}, eps));
  x_minmax->add(epsilon_around_plane_domain({x_axis, in.x_domain_size}, eps));
  in.node_sets["x_minmax"] = std::move(x_minmax);
  in.zero_acceleration_conditions.push_back({"x_minmax", x_axis});
  auto y_minmax = std::make_unique<union_domain>();
  y_minmax->add(epsilon_around_plane_domain({y_axis, 0.0}, eps));
  y_minmax->add(epsilon_around_plane_domain({y_axis, in.y_domain_size}, eps));
  in.node_sets["y_minmax"] = std::move(y_minmax);
  in.zero_acceleration_conditions.push_back({"y_minmax", y_axis});
  run(in);
}

static void LGR_NOINLINE run_elastic_wave_3d() {
  input in;
  in.name = "elastic_wave_3d";
  in.element = TETRAHEDRON;
  in.end_time = 2.0e-3;
  in.num_file_outputs = 100;
  in.elements_along_x = 1000;
  in.elements_along_y = 1;
  in.y_domain_size = 1.0e-3;
  in.elements_along_z = 1;
  in.z_domain_size = 1.0e-3;
  in.rho0 = 1000.0;
  in.enable_neo_Hookean = true;
  in.K0 = 1.0e9;
  in.G0 = 0.0;
  in.initial_v = set_exponential_wave_v;
  static constexpr vector3<double> x_axis(1.0, 0.0, 0.0);
  static constexpr vector3<double> y_axis(0.0, 1.0, 0.0);
  static constexpr vector3<double> z_axis(0.0, 0.0, 1.0);
  static constexpr double eps = 1.0e-10;
  auto x_minmax = std::make_unique<union_domain>();
  x_minmax->add(epsilon_around_plane_domain({x_axis, 0.0}, eps));
  x_minmax->add(epsilon_around_plane_domain({x_axis, in.x_domain_size}, eps));
  in.node_sets["x_minmax"] = std::move(x_minmax);
  in.zero_acceleration_conditions.push_back({"x_minmax", x_axis});
  auto y_minmax = std::make_unique<union_domain>();
  y_minmax->add(epsilon_around_plane_domain({y_axis, 0.0}, eps));
  y_minmax->add(epsilon_around_plane_domain({y_axis, in.y_domain_size}, eps));
  in.node_sets["y_minmax"] = std::move(y_minmax);
  in.zero_acceleration_conditions.push_back({"y_minmax", y_axis});
  auto z_minmax = std::make_unique<union_domain>();
  z_minmax->add(epsilon_around_plane_domain({z_axis, 0.0}, eps));
  z_minmax->add(epsilon_around_plane_domain({z_axis, in.z_domain_size}, eps));
  in.node_sets["z_minmax"] = std::move(z_minmax);
  in.zero_acceleration_conditions.push_back({"z_minmax", z_axis});
  run(in);
}

static void LGR_NOINLINE swinging_cube() {
  input in;
  in.name = "swinging_cube";
  in.element = TETRAHEDRON;
  in.num_file_outputs = 100;
  in.elements_along_x = 8;
  in.x_domain_size = 2.0;
  in.elements_along_y = 8;
  in.y_domain_size = 2.0;
  in.elements_along_z = 8;
  in.z_domain_size = 2.0;
  double const rho = 1.1e3;
  in.rho0 = rho;
  in.enable_neo_Hookean = true;
  double const nu = 0.45;
  double const E = 1.7e7;
  double const K = E / (3.0 * (1.0 - 2.0 * nu));
  double const G = E / (2.0 * (1.0 + nu));
  double const w = pi * std::sqrt((3.0 * G) / (4.0 * rho));
  in.end_time = 0.10;
  in.K0 = K;
  in.G0 = G;
  auto swinging_cube_v = [=] (
    int_range const nodes,
    host_vector<vector3<double>> const& x_vector,
    host_vector<vector3<double>>* v_vector) {
    auto const nodes_to_x = x_vector.cbegin();
    auto const nodes_to_v = v_vector->begin();
    double const U0 = 5.0e-4;
    auto functor = [=](int const node) {
      constexpr double half_pi = pi / 2.0;
      vector3<double> const x = nodes_to_x[node];
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
  in.node_sets["x_min"] = epsilon_around_plane_domain({x_axis, 0.0}, eps);
  in.node_sets["x_max"] = epsilon_around_plane_domain({x_axis, in.x_domain_size}, eps);
  in.node_sets["y_min"] = epsilon_around_plane_domain({y_axis, 0.0}, eps);
  in.node_sets["y_max"] = epsilon_around_plane_domain({y_axis, in.y_domain_size}, eps);
  in.node_sets["z_min"] = epsilon_around_plane_domain({z_axis, 0.0}, eps);
  in.node_sets["z_max"] = epsilon_around_plane_domain({z_axis, in.z_domain_size}, eps);
  in.zero_acceleration_conditions.push_back({"x_min", x_axis});
  in.zero_acceleration_conditions.push_back({"x_max", x_axis});
  in.zero_acceleration_conditions.push_back({"y_min", y_axis});
  in.zero_acceleration_conditions.push_back({"y_max", y_axis});
  in.zero_acceleration_conditions.push_back({"z_min", z_axis});
  in.zero_acceleration_conditions.push_back({"z_max", z_axis});
  in.enable_nodal_pressure = true;
  in.c_tau = 0.5;
  in.CFL = 0.45;
  run(in);
}

static void LGR_NOINLINE bending_beam() {
  input in;
  in.name = "bending_beam";
  in.element = TETRAHEDRON;
  in.num_file_outputs = 100;
  in.elements_along_x = 3;
  in.x_domain_size = 1.0;
  in.elements_along_y = 18;
  in.y_domain_size = 6.0;
  in.elements_along_z = 3;
  in.z_domain_size = 1.0;
  double const rho = 1.1e3;
  in.rho0 = rho;
  in.enable_neo_Hookean = true;
  double const nu = 0.499;
  double const E = 1.7e7;
  double const K = E / (3.0 * (1.0 - 2.0 * nu));
  double const G = E / (2.0 * (1.0 + nu));
  in.end_time = 1.0;
  in.K0 = K;
  in.G0 = G;
  auto bending_beam_v = [=] (
    int_range const nodes,
    host_vector<vector3<double>> const& x_vector,
    host_vector<vector3<double>>* v_vector) {
    auto const nodes_to_x = x_vector.cbegin();
    auto const nodes_to_v = v_vector->begin();
    auto functor = [=](int const node) {
      vector3<double> const x = nodes_to_x[node];
      auto const v = vector3<double>((5.0 / 3.0) * x(1), 0.0, 0.0);
      nodes_to_v[node] = v;
    };
    lgr::for_each(nodes, functor);
  };
  in.initial_v = bending_beam_v;
  static constexpr vector3<double> x_axis(1.0, 0.0, 0.0);
  static constexpr vector3<double> y_axis(0.0, 1.0, 0.0);
  static constexpr vector3<double> z_axis(0.0, 0.0, 1.0);
  static constexpr double eps = 1.0e-10;
  in.node_sets["y_min"] = epsilon_around_plane_domain({y_axis, 0.0}, eps);
  in.zero_acceleration_conditions.push_back({"y_min", x_axis});
  in.zero_acceleration_conditions.push_back({"y_min", y_axis});
  in.zero_acceleration_conditions.push_back({"y_min", z_axis});
  in.enable_nodal_pressure = true;
  in.c_tau = 0.15;
  run(in);
}

static void LGR_NOINLINE twisting_column() {
  input in;
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
  in.rho0 = rho;
  in.enable_neo_Hookean = true;
  double const nu = 0.499;
  double const E = 1.7e7;
  double const K = E / (3.0 * (1.0 - 2.0 * nu));
  double const G = E / (2.0 * (1.0 + nu));
  in.K0 = K;
  in.G0 = G;
  auto twisting_column_v = [=] (
    int_range const nodes,
    host_vector<vector3<double>> const& x_vector,
    host_vector<vector3<double>>* v_vector) {
    auto const nodes_to_x = x_vector.cbegin();
    auto const nodes_to_v = v_vector->begin();
    auto functor = [=](int const node) {
      vector3<double> const x = nodes_to_x[node];
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
  in.node_sets["y_min"] = epsilon_around_plane_domain({y_axis, 0.0}, eps);
  in.zero_acceleration_conditions.push_back({"y_min", x_axis});
  in.zero_acceleration_conditions.push_back({"y_min", y_axis});
  in.zero_acceleration_conditions.push_back({"y_min", z_axis});
  in.enable_nodal_pressure = true;
  in.c_tau = 0.5;
  in.CFL = 0.9;
  run(in);
}

static void LGR_NOINLINE tet_piston() {
  input in;
  in.name = "tet_piston";
  in.element = TETRAHEDRON;
  in.end_time = 10.0;
  in.num_file_outputs = 100;
  in.elements_along_x = 1;
  in.x_domain_size = 1.0;
  in.elements_along_y = 1;
  in.y_domain_size = 1.0;
  in.elements_along_z = 1;
  in.z_domain_size = 1.0;
  double const rho = 1.0;
  in.rho0 = rho;
  in.enable_neo_Hookean = true;
  double const K = 1.0;
  double const G = 1.0;
  in.K0 = K;
  in.G0 = G;
  auto tet_piston_v = [=] (
    int_range const nodes,
    host_vector<vector3<double>> const& x_vector,
    host_vector<vector3<double>>* v_vector) {
    auto const nodes_to_x = x_vector.cbegin();
    auto const nodes_to_v = v_vector->begin();
    auto functor = [=](int const node) {
      vector3<double> const x = nodes_to_x[node];
      auto const v = 0.6 * (x(2) / 1.0) * vector3<double>(0.0, 0.0, 1.0);
      nodes_to_v[node] = v;
    };
    lgr::for_each(nodes, functor);
  };
  in.initial_v = tet_piston_v;
  static constexpr vector3<double> x_axis(1.0, 0.0, 0.0);
  static constexpr vector3<double> y_axis(0.0, 1.0, 0.0);
  static constexpr vector3<double> z_axis(0.0, 0.0, 1.0);
  static constexpr double eps = 1.0e-10;
  in.node_sets["x_min"] = epsilon_around_plane_domain({x_axis, 0.0}, eps);
  in.node_sets["x_max"] = epsilon_around_plane_domain({x_axis, in.x_domain_size}, eps);
  in.node_sets["y_min"] = epsilon_around_plane_domain({y_axis, 0.0}, eps);
  in.node_sets["y_max"] = epsilon_around_plane_domain({y_axis, in.y_domain_size}, eps);
  in.node_sets["z_min"] = epsilon_around_plane_domain({z_axis, 0.0}, eps);
  in.zero_acceleration_conditions.push_back({"x_min", x_axis});
  in.zero_acceleration_conditions.push_back({"x_max", x_axis});
  in.zero_acceleration_conditions.push_back({"y_min", y_axis});
  in.zero_acceleration_conditions.push_back({"y_max", y_axis});
  in.zero_acceleration_conditions.push_back({"z_min", z_axis});
  in.enable_nodal_pressure = true;
  in.c_tau = 0.5;
  in.CFL = 0.9;
  run(in);
}

static void LGR_NOINLINE run_Noh_1D() {
  input in;
  in.name = "Noh_1D";
  in.element = BAR;
  in.end_time = 0.6;
  in.num_file_outputs = 60;
  in.elements_along_x = 44;
  in.x_domain_size = 1.1;
  in.rho0 = 1.0;
  in.enable_ideal_gas = true;
  in.gamma = 5.0 / 3.0;
  in.e0 = 1.0e-14;
  auto inward_v = [=] (
    int_range const nodes,
    host_vector<vector3<double>> const& x_vector,
    host_vector<vector3<double>>* v_vector) {
    auto const nodes_to_x = x_vector.cbegin();
    auto const nodes_to_v = v_vector->begin();
    auto functor = [=](int const node) {
      vector3<double> const x = nodes_to_x[node];
      auto const n = norm(x);
      auto const v = (n == 0) ? vector3<double>::zero() : (-(x / n));
      nodes_to_v[node] = v;
    };
    lgr::for_each(nodes, functor);
  };
  in.initial_v = inward_v;
  static constexpr vector3<double> x_axis(1.0, 0.0, 0.0);
  static constexpr double eps = 1.0e-10;
  in.node_sets["x_min"] = epsilon_around_plane_domain({x_axis, 0.0}, eps);
  in.zero_acceleration_conditions.push_back({"x_min", x_axis});
  in.enable_viscosity = true;
  in.linear_artificial_viscosity = 1.0;
  in.quadratic_artificial_viscosity = 1.0;
  run(in);
}

static void LGR_NOINLINE run_Noh_2D() {
  input in;
  in.name = "Noh_2D";
  in.element = TRIANGLE;
  in.end_time = 0.6;
  in.num_file_outputs = 60;
  in.elements_along_x = 34;
  in.x_domain_size = 0.85;
  in.elements_along_y = 34;
  in.y_domain_size = 0.85;
  in.rho0 = 1.0;
  in.enable_ideal_gas = true;
  in.gamma = 5.0 / 3.0;
  in.e0 = 1.0e-14;
  auto inward_v = [=] (
    int_range const nodes,
    host_vector<vector3<double>> const& x_vector,
    host_vector<vector3<double>>* v_vector) {
    auto const nodes_to_x = x_vector.cbegin();
    auto const nodes_to_v = v_vector->begin();
    auto functor = [=](int const node) {
      vector3<double> const x = nodes_to_x[node];
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
  in.node_sets["x_min"] = epsilon_around_plane_domain({x_axis, 0.0}, eps);
  in.node_sets["y_min"] = epsilon_around_plane_domain({y_axis, 0.0}, eps);
  in.zero_acceleration_conditions.push_back({"x_min", x_axis});
  in.zero_acceleration_conditions.push_back({"y_min", y_axis});
  in.enable_viscosity = true;
  in.linear_artificial_viscosity = 0.5;
  in.quadratic_artificial_viscosity = 1.0;
  in.enable_nodal_energy = true;
  in.c_tau = 1.0;
  run(in);
}

}

int main() {
  if ((0)) lgr::run_elastic_wave();
  if ((0)) lgr::run_gas_expansion();
  if ((0)) lgr::spinning_square();
  if ((0)) lgr::Cooks_membrane();
  if ((0)) lgr::swinging_plate();
  if ((0)) lgr::spinning_cube();
  if ((0)) lgr::run_elastic_wave_2d();
  if ((0)) lgr::run_elastic_wave_3d();
  if ((0)) lgr::swinging_cube();
  if ((0)) lgr::bending_beam();
  if ((0)) lgr::twisting_column();
  if ((0)) lgr::tet_piston();
  if ((0)) lgr::run_Noh_1D();
  if ((0)) lgr::run_Noh_2D();
  if ((1)) {
    lgr::product_range<lgr::counting_iterator<lgr::element_node>, lgr::SOA, lgr::node_in_element, lgr::element>
      range(lgr::counting_iterator<lgr::element_node>(lgr::element_node(0)),
            lgr::element(10),
            lgr::node_in_element(3));
    for (auto const inner : range) {
      for (auto const i : inner) {
        std::cout << int(i) << '\n';
      }
    }
  }
}