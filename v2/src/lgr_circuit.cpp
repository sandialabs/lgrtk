#include <lgr_circuit.hpp>
#include <algorithm>

namespace lgr {

void assemble_circuit(
    std::vector<int> const& resistor_dofs,
    std::vector<int> const& inductor_dofs,
    std::vector<int> const& capacitor_dofs,
    std::vector<double> const& resistances,
    std::vector<double> const& inductances,
    std::vector<double> const& capacitances,
    int const ground_dof,
    MediumMatrix& M,
    MediumMatrix& K
    ) {
  auto const nresistors = int(resistances.size());
  auto const ninductors = int(inductances.size());
  auto const ncapacitors = int(capacitances.size());
  OMEGA_H_CHECK(int(resistor_dofs.size()) == nresistors * 2);
  OMEGA_H_CHECK(int(inductor_dofs.size()) == ninductors * 3);
  OMEGA_H_CHECK(int(capacitor_dofs.size()) == ncapacitors * 2);
  int max_dof = -1;
  if (!resistor_dofs.empty()) max_dof = std::max(max_dof, *std::max_element(resistor_dofs.begin(), resistor_dofs.end()));
  if (!inductor_dofs.empty()) max_dof = std::max(max_dof, *std::max_element(inductor_dofs.begin(), inductor_dofs.end()));
  if (!capacitor_dofs.empty()) max_dof = std::max(max_dof, *std::max_element(capacitor_dofs.begin(), capacitor_dofs.end()));
  int n = max_dof + 1;
  M = MediumMatrix(n);
  K = MediumMatrix(n);
  for (int c = 0; c < nresistors; ++c) {
    auto const i = resistor_dofs[std::size_t(c * 2 + 0)];
    auto const j = resistor_dofs[std::size_t(c * 2 + 1)];
    auto const R = resistances[std::size_t(c)];
    auto const G = 1.0 / R;
    K(i, i) += G * 1.0;
    K(j, j) += G * 1.0;
    K(i, j) += G * -1.0;
    K(j, i) += G * -1.0;
  }
  for (int c = 0; c < ninductors; ++c) {
    auto const i = inductor_dofs[std::size_t(c * 3 + 0)];
    auto const j = inductor_dofs[std::size_t(c * 3 + 1)];
    auto const k = inductor_dofs[std::size_t(c * 3 + 2)];
    auto const L = inductances[std::size_t(c)];
    K(i, k) += 1.0;
    K(j, k) += -1.0;
    K(k, i) += -1.0;
    K(k, j) += 1.0;
    M(k, k) += L;
  }
  for (int c = 0; c < ncapacitors; ++c) {
    auto const i = capacitor_dofs[std::size_t(c * 2 + 0)];
    auto const j = capacitor_dofs[std::size_t(c * 2 + 1)];
    auto const C = capacitances[std::size_t(c)];
    M(i, i) += C * 1.0;
    M(j, j) += C * 1.0;
    M(i, j) += C * -1.0;
    M(j, i) += C * -1.0;
  }
  for (int j = 0; j < n; ++j) {
    M(ground_dof, j) = (ground_dof == j) ? 1.0 : 0.0;
    K(ground_dof, j) = (ground_dof == j) ? 1.0 : 0.0;
  }
}

void form_backward_euler_circuit_system(
    MediumMatrix const& M,
    MediumMatrix const& K,
    MediumVector const& last_x,
    double const dt,
    MediumMatrix& A,
    MediumVector& b)
{
  auto const n = M.size;
  A = MediumMatrix(n);
  b = MediumVector(n);
  auto const inv_dt = 1.0 / dt;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      A(i, j) = inv_dt * M(i, j) + K(i, j);
      b(i) += inv_dt * M(i, j) * last_x(j);
    }
  }
}

}
