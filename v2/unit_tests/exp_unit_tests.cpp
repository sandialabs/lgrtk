#include <lgr_exp.hpp>
#include "lgr_gtest.hpp"

#include <Omega_h_print.hpp>

template <int N>
static bool is_close(
    Omega_h::Tensor<N> a,
    Omega_h::Tensor<N> b,
    double eps = 1e-12) {
  bool close = true;
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      if (! Omega_h::are_close(a(i, j), b(i, j), eps)) {
        close = false;
      }
    }
  }
  return close;
}

TEST(exp, test1) {
  lgr::Tensor<3> A;
  A(0, 0) = 1.0; A(0, 1) = 2.0; A(0, 2) = 3.0;
  A(1, 0) = 4.0; A(1, 1) = 5.0; A(1, 2) = 6.0;
  A(2, 0) = 7.0; A(2, 1) = 8.0; A(2, 2) = 9.0;
  lgr::Tensor<3> Ae;
  Ae(0, 0) = 1.118906699413195e+06;
  Ae(0, 1) = 1.374815062935818e+06;
  Ae(0, 2) = 1.630724426458440e+06;
  Ae(1, 0) = 2.533881041898991e+06;
  Ae(1, 1) = 3.113415031380579e+06;
  Ae(1, 2) = 3.692947020862166e+06;
  Ae(2, 0) = 3.948856384384790e+06;
  Ae(2, 1) = 4.852012999825342e+06;
  Ae(2, 2) = 5.755170615265895e+06;
  EXPECT_TRUE(is_close<3>(lgr::exp::exp(A), Ae));
}

TEST(exp, test2) {
  std::srand(0);
  lgr::Tensor<3> A;
  double const max = static_cast<double>(RAND_MAX);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      A(i, j) = static_cast<double>(std::rand()) / max;
    }
  }
  lgr::Tensor<3> const B = lgr::exp::exp(A);
  lgr::Tensor<3> const C = lgr::exp::exp_taylor(A);
  lgr::Tensor<3> const D = B - C;
  double const error = norm(D) / norm(C);

  EXPECT_TRUE(error <= 64.0 * DBL_EPSILON);
}
