#include <lgr_element_functions.hpp>
#include "lgr_gtest.hpp"

#include <Omega_h_print.hpp>

using Omega_h::are_close;

template <int M, int N>
static bool is_close(
    Omega_h::Matrix<M, N> a,
    Omega_h::Matrix<M, N> b,
    double eps = 1e-10) {
  bool close = true;
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      if (! are_close(a(i, j), b(i, j), eps)) {
        std::cout << a(i, j) << ", " << b(i, j) << std::endl;
        close = false;
      }
    }
  }
  return close;
}

static Omega_h::Matrix<2, 6> get_parametric_coords() {
  Omega_h::Matrix<6, 2> xi;
  xi = {
    0.0, 0.0,
    1.0, 0.0,
    0.0, 1.0,
    0.5, 0.0,
    0.5, 0.5,
    0.0, 0.5 };
  return Omega_h::transpose(xi);
}

static Omega_h::Matrix<2, 6> get_grads(int ip) {
  Omega_h::Matrix<6, 2> dN;
  switch (ip) {
    case 0:
      dN = {
         1.0,   1.0,
         5.0,   0.0,
         0.0,  -1.0,
        -6.0,  -8.0,
         2.0,   8.0,
        -2.0,   0.0 };
      break;
    case 1:
      dN = {
         1.0,   1.0,
        -1.0,   0.0,
         0.0,   5.0,
         0.0,  -2.0,
         8.0,   2.0,
        -8.0,  -6.0 };
      break;
    case 2:
      dN = {
        -5.0,  -5.0,
        -1.0,   0.0,
         0.0,  -1.0,
         6.0,  -2.0,
         2.0,   2.0,
        -2.0,   6.0 };
      break;
    default:
      Omega_h_fail("invalid integration point\n");
  }
  dN /= 3.0;
  return Omega_h::transpose(dN);
}

TEST(tri6, gradient) {
  auto x = get_parametric_coords();
  auto shape = lgr::Tri6::shape(x);
  for (int i = 0; i < lgr::Tri6::points; ++i ) {
    auto g_gold = get_grads(i);
    auto g = shape.basis_gradients[i];
    OMEGA_H_CHECK(is_close(g_gold, g));
  }
}
