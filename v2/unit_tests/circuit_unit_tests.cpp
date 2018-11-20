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
  x(0) = 0.0;
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
    lgr::form_backward_euler_circuit_system(M, K, x, dt, A, b);
    lgr::gaussian_elimination(A, b);
    lgr::back_substitution(A, b, x);
    t += dt;
    double const expected_V = V0 * std::exp(-t / (R * C));
    EXPECT_TRUE(Omega_h::are_close(x(1), expected_V, 0.1, 0.0));
  }
}

LGR_END_TESTS
