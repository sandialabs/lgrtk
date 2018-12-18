#ifndef LGR_CIRCUIT_HPP
#define LGR_CIRCUIT_HPP

#include <lgr_linear_algebra.hpp>

namespace lgr {

// assembles M * udot + K * u = b
void assemble_circuit(std::vector<int> const& resistor_dofs,
    std::vector<int> const& inductor_dofs,
    std::vector<int> const& capacitor_dofs,
    std::vector<double> const& resistances,
    std::vector<double> const& inductances,
    std::vector<double> const& capacitances, int const ground_dof,
    MediumMatrix& M, MediumMatrix& K);

void form_backward_euler_circuit_system(MediumMatrix const& M,
    MediumMatrix const& K, MediumVector const& last_x, double const dt,
    MediumMatrix& A, MediumVector& b);

}  // namespace lgr

#endif
