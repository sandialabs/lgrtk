#include <lgr_circuit.hpp>
#include "lgr_gtest.hpp"
#include <fstream>
#include <cmath>

TEST(circuit, RC) {
  std::ofstream file("circuit.csv");
  int const n = 2;
  std::vector<int> resistor_dofs = {0, 1};
  std::vector<int> inductor_dofs = {};
  std::vector<int> capacitor_dofs = {0, 1};
  lgr::MediumVector x(n);
  double const V0 = 1.0;
  x(0) = 0.0;
  x(1) = V0;
  file << std::scientific << std::setprecision(6);
  for (int j = 0; j < n; ++j) {
    if (j > 0) file << ", ";
    file << x(j);
  }
  file << ", " << V0;
  file << '\n';
  double const R = 1.0;
  double const C = 1.0;
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
  std::cout << "M\n";
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      std::cout << M(i, j) << ' ';
    }
    std::cout << '\n';
  }
  std::cout << "K\n";
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      std::cout << K(i, j) << ' ';
    }
    std::cout << '\n';
  }
  lgr::MediumMatrix A;
  lgr::MediumVector b;
  double const dt = 0.1;
  double t = 0.0;
  for (int s = 0; s < 40; ++s) {
    lgr::form_backward_euler_circuit_system(M, K, x, dt, A, b);
    lgr::gaussian_elimination(A, b);
    lgr::back_substitution(A, b, x);
    for (int j = 0; j < n; ++j) {
      if (j > 0) file << ", ";
      file << x(j);
    }
    t += dt;
    double const expected_V = V0 * std::exp(-t / (R * C));
    file << ", " << expected_V;
    file << '\n';
  }
}

LGR_END_TESTS
