#ifndef LGR_CIRCUIT_HPP
#define LGR_CIRCUIT_HPP

#include <lgr_linear_algebra.hpp>

namespace lgr {

// assembles M * udot + K * u = b
void assemble_circuit(
    std::vector<int> const& resistor_dofs,
    std::vector<int> const& inductor_dofs,
    std::vector<int> const& capacitor_dofs,
    std::vector<double> const& resistances,
    std::vector<double> const& inductances,
    std::vector<double> const& capacitances,
    MediumMatrix& M,
    MediumMatrix& K,
    MediumVector& b
    );

}

#endif
