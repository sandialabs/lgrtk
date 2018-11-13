#include <lgr_linear_algebra.hpp>
#include "lgr_gtest.hpp"
#include <lgr_for.hpp>
#include <Omega_h_int_scan.hpp>
#include <Omega_h_array_ops.hpp>

TEST(linear_algebra, wikipedia_cg) {
  Omega_h::Write<int> offsets = {0, 2, 4};
  Omega_h::Write<int> indices = {0, 1, 0, 1};
  Omega_h::Write<double> entries = {4.0, 1.0, 1.0, 3.0};
  Omega_h::Graph rows_to_columns(offsets, indices);
  lgr::GlobalMatrix A;
  A.rows_to_columns = rows_to_columns;
  A.entries = entries;
  Omega_h::Write<double> b({1.0, 2.0});
  Omega_h::Write<double> computed_x({2.0, 1.0});
  double const tol = 1e-10;
  conjugate_gradient(A, b, computed_x, tol);
  Omega_h::Read<double> known_x({1.0 / 11.0, 7.0 / 11.0});
  OMEGA_H_CHECK(are_close(read(computed_x), known_x, tol, tol));
}

static void run_fd_cg() {
  int const ndofs = 8;
  Omega_h::Write<int> counts(ndofs);
  auto f0 = OMEGA_H_LAMBDA(int const i) {
    if (i == 0 || i == ndofs - 1) counts[i] = 2;
    else counts[i] = 3;
  };
  lgr::parallel_for(ndofs, std::move(f0));
  auto const offsets = Omega_h::offset_scan(read(counts));
  auto const nnz = offsets.last();
  Omega_h::Write<int> indices(nnz);
  auto f1 = OMEGA_H_LAMBDA(int const i) {
    auto nz = offsets[i];
    if (i > 0) indices[nz++] = i - 1;
    indices[nz++] = i;
    if (i < ndofs - 1) indices[nz++] = i + 1;
    OMEGA_H_CHECK(offsets[i + 1] == nz);
  };
  lgr::parallel_for(ndofs, std::move(f1));
  Omega_h::Write<double> values(nnz);
  auto f2 = OMEGA_H_LAMBDA(int const i) {
    auto nz = offsets[i];
    if (i > 0) values[nz++] = -1.0;
    values[nz++] = 2.0;
    if (i < ndofs - 1) values[nz++] = -1.0;
    OMEGA_H_CHECK(offsets[i + 1] == nz);
  };
  lgr::parallel_for(ndofs, std::move(f2));
  Omega_h::Write<double> rhs(ndofs, 0.0);
  double const left_temp = 0.0;
  double const right_temp = 1.0;
  rhs.set(0, rhs.get(0) + left_temp);
  rhs.set(ndofs - 1, rhs.get(ndofs - 1) + right_temp);
  double const dx = 1.0 / double(ndofs + 1);
  double const dtemp = right_temp - left_temp;
  Omega_h::Write<double> known_answer(ndofs, left_temp + dtemp * dx, dx * dtemp);
  Omega_h::Write<double> computed_answer(ndofs, 0.0);
  double const tol = 1e-8;
  Omega_h::Graph rows_to_columns(offsets, indices);
  lgr::GlobalMatrix A;
  A.rows_to_columns = rows_to_columns;
  A.entries = values;
  OMEGA_H_CHECK(0 == lgr::conjugate_gradient(A, rhs, known_answer, tol));
  int const niter = conjugate_gradient(A, rhs, computed_answer, tol);
  OMEGA_H_CHECK(niter <= ndofs);
  OMEGA_H_CHECK(Omega_h::are_close(read(known_answer), read(computed_answer), tol, tol));
}

TEST(linear_algebra, finite_difference_cg) {
  run_fd_cg();
}

LGR_END_TESTS
