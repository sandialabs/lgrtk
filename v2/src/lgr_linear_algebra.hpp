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
void set_boundary_conditions(
    GlobalMatrix A,
    GlobalVector x,
    GlobalVector b,
    Omega_h::LOs rows_to_bc_rows);

struct MediumMatrix {
  int size;
  std::vector<double> entries;
  inline double& operator()(int const i, int const j) noexcept {
    return entries[std::size_t(i * size + j)];
  }
  inline double const& operator()(int const i, int const j) const noexcept {
    return entries[std::size_t(i * size + j)];
  }
  MediumMatrix(int const size_in);
};

struct MediumVector {
  std::vector<double> entries;
  inline double& operator()(int const i) noexcept { return entries[std::size_t(i)]; }
  inline double const& operator()(int const i) const noexcept { return entries[std::size_t(i)]; }
  MediumVector(int const size_in);
  MediumVector() = default;
};

void gaussian_elimination(MediumMatrix& A, MediumVector& b);
void back_substitution(MediumMatrix const& A, MediumVector const& b, MediumVector& x);

}

#endif
