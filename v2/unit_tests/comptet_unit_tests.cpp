#include <lgr_element_functions.hpp>
#include "lgr_gtest.hpp"

using Omega_h::are_close;

template <int M, int N>
static bool is_close(Omega_h::Matrix<M, N> a, Omega_h::Matrix<M, N> b) {
  bool close = true;
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      if (! are_close(a(i, j), b(i, j))) {
        close = false;
      }
    }
  }
  return close;
}

static Omega_h::Matrix<3, 3> I3x3() {
  Omega_h::Matrix<3, 3> I = Omega_h::zero_matrix<3, 3>();
  for (int i = 0; i < 3; ++i) {
    I(i, i) = 1.0;
  }
  return I;
}

static Omega_h::Matrix<3, 10> get_parametric_coords() {
  Omega_h::Matrix<3, 10> xi;
  xi[0][0] = 0.0; xi[0][1] = 0.0; xi[0][2] = 0.0;
  xi[1][0] = 1.0; xi[1][1] = 0.0; xi[1][2] = 0.0;
  xi[2][0] = 0.0; xi[2][1] = 1.0; xi[2][2] = 0.0;
  xi[3][0] = 0.0; xi[3][1] = 0.0; xi[3][2] = 1.0;
  xi[4][0] = 0.5; xi[4][1] = 0.0; xi[4][2] = 0.0;
  xi[5][0] = 0.5; xi[5][1] = 0.5; xi[5][2] = 0.0;
  xi[6][0] = 0.0; xi[6][1] = 0.5; xi[6][2] = 0.0;
  xi[7][0] = 0.0; xi[7][1] = 0.0; xi[7][2] = 0.5;
  xi[8][0] = 0.5; xi[8][1] = 0.0; xi[8][2] = 0.5;
  xi[9][0] = 0.0; xi[9][1] = 0.5; xi[9][2] = 0.5;
  return xi;
}

static Omega_h::Matrix<3, 10> get_reference_coords() {
  Omega_h::Matrix<3, 10> X;
  X[0][0] = -0.05;    X[0][1] = -0.1;     X[0][2] = 0.125;
  X[1][0] = 0.9;      X[1][1] = 0.025;    X[1][2] = -0.075;
  X[2][0] = -0.1;     X[2][1] = 1.025;    X[2][2] = -0.075;
  X[3][0] = 0.125;    X[3][1] = 0.1;      X[3][2] = 1.025;
  X[4][0] = 0.525;    X[4][1] = 0.025;    X[4][2] = -0.05;
  X[5][0] = 0.5;      X[5][1] = 0.525;    X[5][2] = 0.05;
  X[6][0] = 0.125;    X[6][1] = 0.45;     X[6][2] = -0.075;
  X[7][0] = 0.1;      X[7][1] = 0.075;    X[7][2] = 0.575;
  X[8][0] = 0.475;    X[8][1] = -0.075;   X[8][2] = 0.55;
  X[9][0] = -0.075;   X[9][1] = 0.6;      X[9][2] = 0.45;
  return X;
}

static lgr::CompTet::OType gold_O_reference() {
  lgr::CompTet::OType O;
  O[0] = {1.1500000000000001, 0.35, 0.30000000000000004,
    0.25, 1.1, 0.35,
    -0.35, -0.4, 0.8999999999999999};
  O[1] = {0.75, -0.050000000000000044, -0.10000000000000009,
    0., 1., -0.2,
    -0.04999999999999999, 0.2, 1.2000000000000002};
  O[2] = {0.75, -0.45, -0.4,
    0.15000000000000002, 1.15, 0.29999999999999993,
    0.25, 0., 1.05};
  O[3] = {0.75, -0.35, 0.04999999999999999,
    -0.3, 1.05, 0.05000000000000002,
    -0.04999999999999982, -0.2499999999999999, 0.8999999999999999};
  O[4] = {0.8500000000000001, -0.050000000000000044, -0.10000000000000009,
    -0.16666666666666669, 1., -0.2,
    0.20000000000000012, 0.2, 1.2000000000000002};
  O[5] = {0.8500000000000001, -0.24999999999999994, -0.30000000000000004,
    -0.16666666666666669, 1.1833333333333331, -0.01666666666666672,
    0.20000000000000012, 0., 1.};
  O[6] = {0.75, -0.35, -0.30000000000000004,
    -0.3, 1.05, -0.01666666666666672,
    -0.04999999999999982, -0.2499999999999999, 1.};
  O[7] = {0.75, -0.14999999999999997, -0.10000000000000009,
    -0.3, 0.8666666666666667, -0.2,
    -0.04999999999999982, -0.04999999999999988, 1.2000000000000002};
  O[8] = {0.75, -0.050000000000000044, -0.2,
    0.15000000000000002, 1., 0.11666666666666664,
    0.25, 0.2, 1.25};
  O[9] = {0.75, -0.24999999999999994, -0.4,
    0.15000000000000002, 1.1833333333333331, 0.29999999999999993,
    0.25, 0., 1.05};
  O[10] = {0.65, -0.35, -0.4,
    0.016666666666666663, 1.05, 0.29999999999999993,
    5.551115123125783e-17, -0.2499999999999999, 1.05};
  O[11] = {0.65, -0.14999999999999997, -0.2,
    0.016666666666666663, 0.8666666666666667, 0.11666666666666664,
    5.551115123125783e-17, -0.04999999999999988, 1.25};
  return O;
}

static Omega_h::Matrix<4, 4> gold_M_inv_parametric() {
  Omega_h::Matrix<4, 4> M_inv = {
    96.0, -24.0, -24.0, -24.0,
    -24.0, 96.0, -24.0, -24.0,
    -24.0, -24.0, 96.0, -24.0,
    -24.0, -24.0, -24.0, 96.0 };
  return M_inv;
}

static Omega_h::Matrix<4, 4> gold_M_inv_reference() {
  Omega_h::Matrix<4, 4> M_inv = {
    88.06190094134685, -22.157433523004897, -20.91267757241373, -25.784431612729065,
    -22.157433523004897, 103.03437794530312, -25.450695338505415, -31.249163121001295,
    -20.912677572413727, -25.45069533850542, 95.7216646218602, -29.950895484185008,
    -25.78443161272906, -31.249163121001292, -29.95089548418501, 131.22814250271662 };
  return M_inv;
}


TEST(composite_tet, O_parametric) {
  auto I = I3x3();
  auto X = get_parametric_coords();
  auto S = lgr::CompTet::compute_S();
  auto O = lgr::CompTet::compute_O(X, S);
  for (int tet = 0; tet < lgr::CompTet::nsub_tets; ++tet) {
    auto O_subtet = O[tet];
    EXPECT_TRUE(is_close(O_subtet, I));
  }
}

TEST(composite_tet, O_reference) {
  auto X = get_reference_coords();
  auto S = lgr::CompTet::compute_S();
  auto O = lgr::CompTet::compute_O(X, S);
  auto O_gold = gold_O_reference();
  for (int tet = 0; tet < lgr::CompTet::nsub_tets; ++tet) {
    auto O_subtet = O[tet];
    auto O_gold_subtet = O_gold[tet];
    EXPECT_TRUE(is_close(O_subtet, O_gold_subtet));
  }
}

TEST(composite_tet, M_inv_parametric) {
  auto X = get_parametric_coords();
  auto S = lgr::CompTet::compute_S();
  auto O = lgr::CompTet::compute_O(X, S);
  auto O_det = lgr::CompTet::compute_O_det(O);
  auto M_inv = lgr::CompTet::compute_M_inv(O_det);
  auto M_inv_gold = gold_M_inv_parametric();
  EXPECT_TRUE(is_close(M_inv, M_inv_gold));
}

TEST(composite_tet, M_inv_reference) {
  auto X = get_reference_coords();
  auto S = lgr::CompTet::compute_S();
  auto O = lgr::CompTet::compute_O(X, S);
  auto O_det = lgr::CompTet::compute_O_det(O);
  auto M_inv = lgr::CompTet::compute_M_inv(O_det);
  auto M_inv_gold = gold_M_inv_reference();
  EXPECT_TRUE(is_close(M_inv, M_inv_gold));
}

TEST(composite_tet, B_parametric) {
  auto I = I3x3();
  auto X = get_parametric_coords();
  auto shape = lgr::CompTet::shape(X);
  for (int pt = 0; pt < lgr::CompTet::points; ++pt) {
    auto BT = shape.basis_gradients[pt];
    auto B = Omega_h::transpose(BT);
    auto result = X * B;
    EXPECT_TRUE(is_close(result, I));
  }
}

TEST(composite_tet, B_reference) {
  auto I = I3x3();
  auto X = get_reference_coords();
  auto shape = lgr::CompTet::shape(X);
  for (int pt = 0; pt < lgr::CompTet::points; ++pt) {
    auto BT = shape.basis_gradients[pt];
    auto B = Omega_h::transpose(BT);
    auto result = X * B;
    EXPECT_TRUE(is_close(result, I));
  }
}
