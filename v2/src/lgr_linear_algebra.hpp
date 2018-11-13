#ifndef LGR_LINEAR_ALGEBRA_HPP
#define LGR_LINEAR_ALGEBRA_HPP

#include <Omega_h_graph.hpp>

namespace lgr {

using GlobalVector = Omega_h::Write<double>;

struct GlobalMatrix {
  Omega_h::Graph rows_to_columns;
  Omega_h::Write<double> entries;
};

void matvec(GlobalMatrix mat, GlobalVector vec, GlobalVector result);
double dot(GlobalVector a, GlobalVector b);
void axpy(double a, GlobalVector x, GlobalVector y, GlobalVector result);
int conjugate_gradient(
    GlobalMatrix A,
    GlobalVector b,
    GlobalVector x,
    double max_residual_magnitude);

}

#endif
