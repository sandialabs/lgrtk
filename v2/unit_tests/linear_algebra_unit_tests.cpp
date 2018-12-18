#include <lgr_linear_algebra.hpp>
#include "lgr_gtest.hpp"
#include <lgr_for.hpp>
#include <Omega_h_int_scan.hpp>
#include <Omega_h_array_ops.hpp>
#include <Omega_h_map.hpp>
#include <Omega_h_print.hpp>

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
  EXPECT_TRUE(are_close(read(computed_x), known_x, tol, tol));
}

static void run_fd_cg() {
  int const ndofs = 3;
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
#ifdef OMEGA_H_USE_CUDA
    OMEGA_H_CHECK(offsets[i + 1] == nz);
#else
    EXPECT_TRUE(offsets[i + 1] == nz);
#endif
  };
  lgr::parallel_for(ndofs, std::move(f1));
  Omega_h::Write<double> values(nnz);
  auto f2 = OMEGA_H_LAMBDA(int const i) {
    auto nz = offsets[i];
    if (i > 0) values[nz++] = -1.0;
    values[nz++] = 2.0;
    if (i < ndofs - 1) values[nz++] = -1.0;
#ifdef OMEGA_H_USE_CUDA
    OMEGA_H_CHECK(offsets[i + 1] == nz);
#else
    EXPECT_TRUE(offsets[i + 1] == nz);
#endif
  };
  lgr::parallel_for(ndofs, std::move(f2));
  Omega_h::Graph rows_to_columns(offsets, indices);
  lgr::GlobalMatrix A;
  A.rows_to_columns = rows_to_columns;
  A.entries = values;
  Omega_h::Write<double> rhs(ndofs, 0.0);
  double const left_temp = 0.0;
  double const right_temp = 1.0;
  double const dx = 1.0 / double(ndofs + 1);
  double const dtemp = right_temp - left_temp;
  Omega_h::Write<double> known_answer(ndofs, left_temp, dx * dtemp);
  Omega_h::LOs bc_rows = {0, ndofs - 1};
  auto rows_to_bc_rows = Omega_h::invert_injective_map(bc_rows, ndofs);
  set_boundary_conditions(A, known_answer, rhs, rows_to_bc_rows);
  Omega_h::Write<double> computed_answer(ndofs, 0.0);
  double const tol = 1e-8;
  EXPECT_TRUE(0 == lgr::conjugate_gradient(A, rhs, known_answer, tol));
  int const niter = conjugate_gradient(A, rhs, computed_answer, tol);
  EXPECT_TRUE(niter <= ndofs);

  EXPECT_TRUE(Omega_h::are_close(read(known_answer), read(computed_answer), tol, tol) );
}

TEST(linear_algebra, finite_difference_cg) {
  run_fd_cg();
}

TEST(linear_algebra, gaussian_elimination_pivot) {
  lgr::MediumMatrix A(3);
  lgr::MediumVector b(3);
  A(0,0) = 1.0;
  A(0,1) =-1.0;
  A(0,2) = 2.0;
  A(1,0) = 0.0;
  A(1,1) = 0.0;
  A(1,2) =-1.0;
  A(2,0) = 0.0;
  A(2,1) = 2.0;
  A(2,2) =-1.0;
  b(0) = 8.0;
  b(1) =-11.0;
  b(2) =-3.0;
  lgr::gaussian_elimination(A, b);
  EXPECT_TRUE(Omega_h::are_close(A(0,0), 1.0));
  EXPECT_TRUE(Omega_h::are_close(A(0,1), -1.0));
  EXPECT_TRUE(Omega_h::are_close(A(0,2), 2.0));
  EXPECT_TRUE(Omega_h::are_close(A(1,0), 0.0));
  EXPECT_TRUE(Omega_h::are_close(A(1,1), 2.0));
  EXPECT_TRUE(Omega_h::are_close(A(1,2), -1.0));
  EXPECT_TRUE(Omega_h::are_close(A(2,0), 0.0));
  EXPECT_TRUE(Omega_h::are_close(A(2,1), 0.0));
  EXPECT_TRUE(Omega_h::are_close(A(2,2), -1.0));
  EXPECT_TRUE(Omega_h::are_close(b(0), 8.0));
  EXPECT_TRUE(Omega_h::are_close(b(1), -3.0));
  EXPECT_TRUE(Omega_h::are_close(b(2), -11.0));
}

TEST(linear_algebra, gaussian_elimination_solve) {
  lgr::MediumMatrix A(3);
  lgr::MediumVector b(3);
  A(0,0) = 3.0;
  A(0,1) = 2.0;
  A(0,2) =-1.0;
  A(1,0) = 2.0;
  A(1,1) =-2.0;
  A(1,2) = 4.0;
  A(2,0) =-1.0;
  A(2,1) = 0.5;
  A(2,2) =-1.0;
  b(0) = 1.0;
  b(1) =-2.0;
  b(2) = 0.0;
  lgr::gaussian_elimination(A, b);
  lgr::MediumVector x;
  lgr::back_substitution(A, b, x);
  EXPECT_TRUE(Omega_h::are_close(x(0), 1.0));
  EXPECT_TRUE(Omega_h::are_close(x(1), -2.0));
  EXPECT_TRUE(Omega_h::are_close(x(2), -2.0));
}

LGR_END_TESTS
