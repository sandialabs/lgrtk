#include <lgr_circuit.hpp>
#include <Omega_h_scalar.hpp>
#include "lgr_gtest.hpp"
#include <fstream>
#include <cmath>

TEST(circuit, RC) {
  int const n = 2;
  std::vector<int> resistor_dofs = {0, 1};
  std::vector<int> inductor_dofs = {};
  std::vector<int> capacitor_dofs = {0, 1};
  lgr::MediumVector x(n);
  double const V0 = 1.0;
  x(1) = V0;
  double const R = 0.7;
  double const C = 0.8;
  std::vector<double> resistances = {R};
  std::vector<double> inductances = {};
  std::vector<double> capacitances = {C};
  int const ground_dof = 0;
  lgr::MediumMatrix M;
  lgr::MediumMatrix K;
  lgr::assemble_circuit(resistor_dofs, inductor_dofs, capacitor_dofs,
      resistances, inductances, capacitances,
      ground_dof,
      M, K);
  lgr::MediumMatrix A;
  lgr::MediumVector b;
  double const dt = 0.05;
  double t = 0.0;
  for (int s = 0; s < 25; ++s) {
    t += dt;
    lgr::form_backward_euler_circuit_system(M, K, x, dt, A, b);
    lgr::gaussian_elimination(A, b);
    lgr::back_substitution(A, b, x);
    double const expected_V = V0 * std::exp(-t / (R * C));
    EXPECT_TRUE(Omega_h::are_close(x(1), expected_V, 0.1, 0.0));
  }
}

TEST(circuit, RLC_underdamped) {
  int const n = 2;
  std::vector<int> resistor_dofs = {1, 2};
  std::vector<int> inductor_dofs = {2, 0, 3};
  std::vector<int> capacitor_dofs = {0, 1};
  lgr::MediumVector x(n);
  double const V0 = 1.0;
  x(1) = V0;
  double const R = 0.7;
  double const L = 1.0;
  double const C = 0.8;
//double const I0 = V0 / R;
  double const damping_factor = (R / 2.0) * std::sqrt(C / L);
  EXPECT_LT(damping_factor, 1.0);
//double const w0 = 1.0 / std::sqrt(L * C);
//double const alpha = R / (2.0 * L);
//double const wd = w0 * std::sqrt(1.0 - Omega_h::square(damping_factor));
//double const phi = Omega_h::PI / 2.0;
  std::vector<double> resistances = {R};
  std::vector<double> inductances = {L};
  std::vector<double> capacitances = {C};
  int const ground_dof = 0;
  lgr::MediumMatrix M;
  lgr::MediumMatrix K;
  lgr::assemble_circuit(resistor_dofs, inductor_dofs, capacitor_dofs,
      resistances, inductances, capacitances,
      ground_dof,
      M, K);
  lgr::MediumMatrix A;
  lgr::MediumVector b;
  double const dt = 0.05;
  double t = 0.0;
  std::ofstream file("circuit.csv");
  file << std::scientific << std::setprecision(8);
//file << x(1) << ", " << V0 << '\n';
  file << x(1) << '\n';
  for (int s = 0; s < 200; ++s) {
    t += dt;
    lgr::form_backward_euler_circuit_system(M, K, x, dt, A, b);
    lgr::gaussian_elimination(A, b);
    lgr::back_substitution(A, b, x);
//  double const expected_I = I0 * std::exp(-alpha * t) * std::sin(wd * t + phi);
//  double const expected_V = expected_I * R;
//  file << x(1) << ", " << expected_V << '\n';
    file << x(1) << '\n';
  }
}

LGR_END_TESTS
