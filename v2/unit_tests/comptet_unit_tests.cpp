#include <lgr_element_functions.hpp>
#include <lgr_stvenant_kirchhoff.hpp>
#include "lgr_gtest.hpp"

#include <Omega_h_print.hpp>

using Omega_h::are_close;

template <int M, int N>
static bool is_close(
    Omega_h::Matrix<M, N> a,
    Omega_h::Matrix<M, N> b,
    double eps = 1e-12) {
  bool close = true;
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      if (! are_close(a(i, j), b(i, j), eps)) {
        close = false;
      }
    }
  }
  return close;
}

template <int N>
static bool is_close(
    Omega_h::Vector<N> a,
    Omega_h::Vector<N> b,
    double eps = 1e-12) {
  bool close = true;
  for (int i = 0; i < N; ++i) {
    if (! are_close(a[i], b[i], eps)) {
      close = false;
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

static Omega_h::Matrix<3, 3> to_sierra_full(Omega_h::Matrix<3, 3> A) {
  Omega_h::Matrix<3, 3> B;
  B(0, 0) = A(0, 0); B(0, 1) = A(1, 1); B(0, 2) = A(2, 2);
  B(1, 0) = A(0, 1); B(1, 1) = A(1, 2); B(1, 2) = A(2, 0);
  B(2, 0) = A(1, 0); B(2, 1) = A(2, 1); B(2, 2) = A(0, 2);
  return B;
}

static Omega_h::Vector<6> to_sierra_symm(Omega_h::Matrix<3, 3> A) {
  Omega_h::Vector<6> B;
  OMEGA_H_CHECK(are_close(A(0, 1), A(1, 0)));
  OMEGA_H_CHECK(are_close(A(0, 2), A(2, 0)));
  OMEGA_H_CHECK(are_close(A(1, 2), A(2, 1)));
  B[0] = A(0, 0);
  B[1] = A(1, 1);
  B[2] = A(2, 2);
  B[3] = A(0, 1);
  B[4] = A(1, 2);
  B[5] = A(2, 0);
  return B;
}

static Omega_h::Matrix<3, 10> get_parametric_coords() {
  Omega_h::Matrix<10, 3> xi;
  xi = {
    0.0, 0.0, 0.0,
    1.0, 0.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 0.0, 1.0,
    0.5, 0.0, 0.0,
    0.5, 0.5, 0.0,
    0.0, 0.5, 0.0,
    0.0, 0.0, 0.5,
    0.5, 0.0, 0.5,
    0.0, 0.5, 0.5 };
  return Omega_h::transpose(xi);
}

static Omega_h::Matrix<3, 10> get_reference_coords() {
  Omega_h::Matrix<10, 3> X;
  X = {
    -0.05,  -0.1,   0.125,
    0.9,    0.025,  -0.075,
    -0.1,   1.025,  -0.075,
    0.125,  0.1,    1.025,
    0.525,  0.025,  -0.05,
    0.5,    0.525,  0.05,
    0.125,  0.45,   -0.075,
    0.1,    0.075,  0.575,
    0.475,  -0.075, 0.55,
    -0.075, 0.6,    0.45 };
  return Omega_h::transpose(X);
}

static Omega_h::Matrix<3, 10> get_current_coords() {
  Omega_h::Matrix<10, 3> x;
  x = {
    -0.05015, -0.10022500000000001, 0.122625,
    0.9729, 0.02979375, -0.09766875,
    -0.1101, 1.1907937499999999, -0.10066875,
    0.1485625, 0.13255, 1.4250062499999998,
    0.5499375, 0.02780625, -0.058925000000000005,
    0.5605, 0.6475875, 0.06973750000000001,
    0.1318125, 0.494325, -0.08975625,
    0.1111, 0.08949375, 0.71113125,
    0.5350874999999999, -0.09320624999999999, 0.7161000000000001,
    -0.0852375, 0.73455, 0.6042375,
  };
  return Omega_h::transpose(x);
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

static Omega_h::Few<Omega_h::Matrix<10, 3>, 4> gold_B_parametric() {
  Omega_h::Few<Omega_h::Matrix<10, 3>, 4> B;
  B[0] = {
    -1.0885254915624212, -1.0885254915624212, -1.0885254915624212,
    -0.029508497187473802, 0., 0.,
    0., -0.02950849718747367, 0.,
    0., 0., -0.02950849718747367,
    1.3043729868748781, -0.5636610018750173, -0.5636610018750172,
    0.37732200375003533, 0.37732200375003544, 0.18633899812498256,
    -0.5636610018750176, 1.3043729868748781, -0.5636610018750173,
    -0.5636610018750176, -0.5636610018750177, 1.3043729868748781,
    0.37732200375003533, 0.1863389981249825, 0.37732200375003483,
    0.18633899812498256, 0.3773220037500351, 0.3773220037500353 };
  B[1] = {
    0.029508497187473726, 0.029508497187473726, 0.029508497187473726,
    1.0885254915624212, 0., 0.,
    0., -0.029508497187473726, 0.,
    0., 0., -0.02950849718747367,
    -1.3043729868748768, -1.8680339887498947, -1.8680339887498947,
    0.5636610018750178, 1.8680339887498953, 5.551115123125783e-17,
    -0.37732200375003533, 4.440892098500626e-16, -0.1909830056250525,
    -0.37732200375003555, -0.1909830056250526, 4.440892098500626e-16,
    0.5636610018750179, 0., 1.8680339887498951,
    -0.18633899812498256, 0.19098300562505288, 0.19098300562505266 };
  B[2] = {
    0.029508497187473726, 0.029508497187473726, 0.029508497187473726,
    -0.02950849718747385, 0., 0.,
    0., 1.088525491562422, 0.,
    0., 0., -0.029508497187473948,
    8.881784197001252e-16, -0.3773220037500351, -0.19098300562505222,
    1.8680339887498962, 0.5636610018750181, 1.6653345369377348e-16,
    -1.8680339887498958, -1.3043729868748777, -1.8680339887498953,
    -0.19098300562505266, -0.3773220037500351, 8.881784197001252e-16,
    0.1909830056250525, -0.18633899812498256, 0.19098300562505224,
    5.551115123125783e-17, 0.5636610018750179, 1.8680339887498962 };
  B[3] = {
    0.029508497187473726, 0.029508497187473726, 0.029508497187473726,
    -0.02950849718747382, 0., 0.,
    0., -0.029508497187473948, 0.,
    0., 0., 1.0885254915624218,
    8.881784197001252e-16, -0.19098300562505233, -0.3773220037500351,
    0.1909830056250525, 0.19098300562505294, -0.1863389981249825,
    -0.1909830056250525, 8.881784197001252e-16, -0.3773220037500348,
    -1.8680339887498953, -1.8680339887498956, -1.3043729868748772,
    1.8680339887498962, 1.1102230246251565e-16, 0.5636610018750177,
    5.551115123125783e-17, 1.8680339887498965, 0.5636610018750174 };
  return B;
}

static Omega_h::Few<Omega_h::Matrix<10, 3>, 4> gold_B_reference() {
  Omega_h::Few<Omega_h::Matrix<10, 3>, 4> B;
  B[0] = {
    -1.0648824950666589, -1.0399437949480013, -0.640924593049488,
    -0.018664096396885058, -0.0006020676257059647, -0.0016556859706914177,
    0.0014179073504135935, -0.015253245739297594, 0.004898225392337863,
    0.0011384880198377773, 0.0018305101495430798, 0.006094259400308111,
    1.3392452775560804, -0.8722563474196265, -0.5012017528993882,
    0.4209992752085011, 0.39578371289233083, 0.16369100464323705,
    -0.9305464535486132, 1.169517951697121, -0.8050215736397817,
    -0.42891924375489504, -0.2910852712370233, 1.2050898962095027,
    0.47304675460311896, 0.25209578782020536, 0.3150060352063141,
    0.2071645860291014, 0.3999127644104547, 0.2540241847076482 };
  B[1] = {
    0.018984779835164556, 0.018540171408111816, 0.011426436575262822,
    1.4420329787297366, 0.046517192862249625, 0.1279222803711864,
    0.003951054020665663, -0.04250376294958502, 0.013649095707754091,
    -0.003877228568955747, -0.006233975346164142, -0.020754576457351348,
    -1.743528190523818, -1.679764944007974, -1.9149700293205414,
    0.7878941834497728, 1.8728651886534406, 0.32662147042377887,
    -0.40275421819072654, 0.05927573833313504, -0.2185891438494107,
    -0.5285039068897748, -0.26716092328271823, -0.13327376778177147,
    0.6555167899662965, -0.14504376271545202, 1.650359397524035,
    -0.2297162418283609, 0.14350907704495575, 0.15760883680705584 };
  B[2] = {
    0.018464278841961113, 0.018031860133719047, 0.01111316080183622,
    -0.05613344841798203, -0.0018107564005800617, -0.00497958010159519,
    -0.08827132906359149, 0.9495855096234842, -0.30493731858331596,
    -0.003751493876278121, -0.006031813683427589, -0.020081526043606468,
    0.14135907370507228, -0.282278726371028, -0.07950846474675,
    2.187936155511044, 1.1507380168180394, 0.4931115295810625,
    -1.6944235744124883, -1.6855740318057926, -1.878816778955477,
    -0.21730512291293197, -0.3800957866219834, -0.022045356231889635,
    0.15790254279184804, -0.11348156975137227, 0.2631952296250232,
    -0.4457770821666528, 0.35091729805894134, 1.542949104654712 };
  B[3] = {
    0.0051636024256369595, 0.005042674968362526, 0.0031078356519642636,
    -0.05138982280650779, -0.0016577362195647652, -0.004558774603803117,
    0.003554092155000809, -0.03823341560682669, 0.012277772899093667,
    0.19509069268300508, 0.3136752313726753, 1.0443090020090293,
    -0.07331027267952006, -0.27292868112926216, -0.4223527675932898,
    0.42979882880380693, 0.3283088860116344, -0.07278268268201485,
    -0.31660739714847896, -0.17324430247401867, -0.4915805285062043,
    -3.487492198468569, -3.0714436875576245, -1.3760082067494905,
    2.6936930039094498, 0.8816917260243936, 0.782166360452026,
    0.6014994711261776, 2.0287893046102314, 0.5254219891226886 };
  return B;
}

static Omega_h::Few<Omega_h::Matrix<3, 3>, 4> gold_F() {
  Omega_h::Few<Omega_h::Matrix<3, 3>, 4> F;
  F[0] = {
    1.0728825078630875, 1.1406706675894112, 1.279007866780825,
    0.018292305243103095, 0.030034273856605998, 0.06559558024841416,
    0.027588238952382906, 0.06587508507237835, 0.02169379108735753 };
  F[1] = {
    1.142308386101098, 1.238550149839732, 1.2954752276290087,
    0.06234376636852145, 0.016402667380209877, 0.004173266751211203,
    0.030775625400816256, 0.010255522860204667, 0.07528694102066706 };
  F[2] = {
    1.132699791580191, 1.2665601355454923, 1.316744081195716,
    0.02659930111638392, 0.12820339271728204, -0.00319003423285983,
    0.11247111175831231, 0.00684797021170469, 0.01706718193717734 };
  F[3] = {
    1.1398198659509842, 1.2480683558047545, 1.4507376580123266,
    0.017647722722288418, 0.040620195033729184, 0.15834784353277337,
    0.026483416170151354, 0.17919629838676054, 0.032210780190121864 };
  return F;
}

static Omega_h::Few<Omega_h::Matrix<3, 3>, 4> gold_F_vol_avg() {
  Omega_h::Few<Omega_h::Matrix<3, 3>, 4> F;
  F[0] = {
    1.1270921320202323, 1.1983054297594666, 1.343632404178044,
    0.019216562079176835, 0.03155181812232642, 0.06890993361476634,
    0.028982192208033372, 0.0692035610023348, 0.022789915083014423 };
  F[1] = {
    1.1381102081571532, 1.2339982670165006, 1.2907141354460807,
    0.06211464240506697, 0.016342384782922385, 0.004157929284864714,
    0.030662519733957054, 0.010217832066370712, 0.07501024868512461 };
  F[2] = {
    1.1175750189621239, 1.2496479455727993, 1.2991617924266083,
    0.026244124586666754, 0.12649151179514598, -0.003147438178040642,
    0.11096930165461966, 0.006756530279326104, 0.016839286383607155 };
  F[3] = {
    1.0936518381941975, 1.1975157586668845, 1.3919760076290466,
    0.016932907533656037, 0.038974887430462735, 0.1519340163537314,
    0.025410714132432476, 0.17193801141969592, 0.030906093161663584 };
  return F;
}

static Omega_h::Vector<4> gold_J() {
  Omega_h::Vector<4> J;
  J = {
    1.0401993083640124,
    0.9375381665845295,
    1.0209015758706936,
    0.6857359491807643 };
  return J;
}

static Omega_h::Few<Omega_h::Vector<6>, 4> gold_cauchy_stress() {
  Omega_h::Few<Omega_h::Vector<6>, 4> sigma;
  sigma[0] = {
    1.8499367316971824e9, 2.2629279309463353e9, 3.3423666918205996e9,
    1.4873843529485816e8, 3.666915013019802e8, 3.041371142825582e8 };
  sigma[1] = {
    1.903761446358733e9, 2.4500891276475005e9, 2.8432014691660953e9,
    2.869156905812986e8, 1.0210315830999057e8, 2.7071030547170925e8 };
  sigma[2] = {
    1.8117087402900019e9, 2.7020007963490562e9, 2.965185931206969e9,
    4.087629090300652e8, 4.9638992732985586e8, 6.804146491430771e7 };
  sigma[3] = {
    1.8017847912199237e9, 2.414307850515131e9, 4.134615467152106e9,
    1.6227216144218203e8, 8.128290771293167e8, 6.204725659982855e8 };
  return sigma;
}

static Omega_h::Few<Omega_h::Matrix<3, 3>, 4> gold_first_pk_stress() {
  Omega_h::Few<Omega_h::Matrix<3, 3>, 4> first_pk;
  first_pk[0] = {
    2.9627833040353236e9, 3.4051519034111733e9, 4.45972095644526e9,
    1.4638379297356075e8, 3.095567251387963e8,  3.908808750632892e8,
    1.7450473161761922e8, 4.269056051310349e8,  2.5014549902738482e8 };
  first_pk[1] = {
    2.9843116170476804e9, 3.58535550141769e9,   3.9851858428708878e9,
    3.4175510386132973e8, 1.1396027797916356e8, 1.6273032901840755e8,
    2.5303532228481236e8, 9.291718288402088e7,  3.6724304895006657e8 };
  first_pk[2] = {
    2.9246551235958343e9, 3.7948703908662357e9, 4.1289994493808618e9,
    3.221074426430058e8,  6.730887040601448e8,  4.098516523361694e7,
    5.626590800086402e8,  2.9727482276880985e8, 1.0019072433257125e8 };
  first_pk[3] = {
    2.9658276050033693e9, 3.6252663393705006e9, 5.152249942248564e9,
    1.672489755254337e8,  5.875929632359095e8,  8.649810850498325e8,
    1.9578312215072435e8, 1.0423197780807687e9, 4.6229749303606486e8 };
  return first_pk;
}

TEST(composite_tet, O_parametric) {
  auto I = I3x3();
  auto X = get_parametric_coords();
  auto S = lgr::CompTet::compute_S();
  auto O = lgr::CompTet::compute_O(X, S);
  for (int tet = 0; tet < lgr::CompTet::nsub_tets; ++tet) {
    EXPECT_TRUE(is_close(O[tet], I));
  }
}

TEST(composite_tet, O_reference) {
  auto X = get_reference_coords();
  auto S = lgr::CompTet::compute_S();
  auto O = lgr::CompTet::compute_O(X, S);
  auto O_gold = gold_O_reference();
  for (int tet = 0; tet < lgr::CompTet::nsub_tets; ++tet) {
    EXPECT_TRUE(is_close(O[tet], O_gold[tet]));
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
  auto B_gold = gold_B_parametric();
  for (int pt = 0; pt < lgr::CompTet::points; ++pt) {
    auto BT = shape.basis_gradients[pt];
    auto B = Omega_h::transpose(BT);
    auto result = X * B;
    EXPECT_TRUE(is_close(result, I));
    EXPECT_TRUE(is_close(B, B_gold[pt]));
  }
}

TEST(composite_tet, B_reference) {
  auto I = I3x3();
  auto X = get_reference_coords();
  auto shape = lgr::CompTet::shape(X);
  auto B_gold = gold_B_reference();
  for (int pt = 0; pt < lgr::CompTet::points; ++pt) {
    auto BT = shape.basis_gradients[pt];
    auto B = Omega_h::transpose(BT);
    auto result = X * B;
    EXPECT_TRUE(is_close(result, I));
    EXPECT_TRUE(is_close(B, B_gold[pt]));
  }
}

TEST(composite_tet, composite_J) {
  auto X = get_reference_coords();
  auto shape = lgr::CompTet::shape(X);
  auto J_gold = gold_J();
  auto ip_weight = 1.0 / 24.0;
  auto w_gold = J_gold * ip_weight;
  for (int pt = 0; pt < lgr::CompTet::points; ++pt) {
    EXPECT_TRUE(are_close(shape.weights[pt], w_gold[pt]));
  }
}

TEST(composite_tet, volume) {
  double volume = 0.0;
  double volume_gold = 0.153515625;
  auto X = get_reference_coords();
  auto shape = lgr::CompTet::shape(X);
  for (int pt = 0; pt < lgr::CompTet::points; ++pt) {
    volume += shape.weights[pt];
  }
  EXPECT_TRUE(are_close(volume, volume_gold));
}

TEST(composite_tet, def_grad) {
  auto X = get_reference_coords();
  auto x = get_current_coords();
  auto shape = lgr::CompTet::shape(X);
  auto F_gold = gold_F();
  for (int pt = 0; pt < lgr::CompTet::points; ++pt) {
    auto BT = shape.basis_gradients[pt];
    auto B = Omega_h::transpose(BT);
    auto F = x * B;
    auto F_sierra = to_sierra_full(F);
    EXPECT_TRUE(is_close(F_sierra, F_gold[pt]));
  }
}

static void do_vol_avg_F(
    Omega_h::Vector<4> weights,
    Omega_h::Few<Omega_h::Matrix<3, 3>, 4>& F) {
  double vol = 0.0;
  double J_bar = 0.0;
  for (int pt = 0; pt < lgr::CompTet::points; ++pt) {
    vol += weights[pt];
    J_bar += weights[pt] * Omega_h::determinant(F[pt]);
  }
  J_bar /= vol;
  for (int pt = 0; pt < lgr::CompTet::points; ++pt) {
    double fac = std::cbrt(J_bar / Omega_h::determinant(F[pt]));
    F[pt] *= fac;
  }
}

TEST(composite_tet, def_grad_vol_avg) {
  auto X = get_reference_coords();
  auto x = get_current_coords();
  auto shape = lgr::CompTet::shape(X);
  Omega_h::Few<Omega_h::Matrix<3, 3>, 4> F_ips;
  for (int pt = 0; pt < lgr::CompTet::points; ++pt) {
    auto BT = shape.basis_gradients[pt];
    auto B = Omega_h::transpose(BT);
    F_ips[pt] = x * B;
  }
  do_vol_avg_F(shape.weights, F_ips);
  auto F_gold_vol_avg = gold_F_vol_avg();
  for (int pt = 0; pt < lgr::CompTet::points; ++pt) {
    auto F_sierra = to_sierra_full(F_ips[pt]);
    EXPECT_TRUE(is_close(F_sierra, F_gold_vol_avg[pt]));
  }
}

static Omega_h::Matrix<3, 3> compute_sigma(Omega_h::Matrix<3, 3> F) {
  double E = 3.45e+9;
  double nu = 0.35;
  double K = E / (3.0 * (1.0 - 2.0 * nu));
  double mu = E / (2.0 * (1.0 + nu));
  double unused = 0.0;
  Omega_h::Matrix<3, 3> cauchy_stress;
  lgr::stvenant_kirchhoff_update(K, mu, 1.0, F, cauchy_stress, unused);
  return cauchy_stress;
}

TEST(composite_tet, cauchy_stress_vol_avg) {
  auto X = get_reference_coords();
  auto x = get_current_coords();
  auto shape = lgr::CompTet::shape(X);
  Omega_h::Few<Omega_h::Matrix<3, 3>, 4> F_ips;
  for (int pt = 0; pt < lgr::CompTet::points; ++pt) {
    auto BT = shape.basis_gradients[pt];
    auto B = Omega_h::transpose(BT);
    F_ips[pt] = x * B;
  }
  do_vol_avg_F(shape.weights, F_ips);
  auto cauchy_stress_gold = gold_cauchy_stress();
  for (int pt = 0; pt < lgr::CompTet::points; ++pt) {
    auto cauchy_stress = compute_sigma(F_ips[pt]);
    auto cauchy_stress_sierra = to_sierra_symm(cauchy_stress);
    EXPECT_TRUE(is_close(cauchy_stress_sierra, cauchy_stress_gold[pt]));
  }
}

static Omega_h::Matrix<3, 3> compute_intermediate_first_PK(
    Omega_h::Matrix<3, 3> F,
    Omega_h::Matrix<3, 3> sigma) {
  auto J = Omega_h::determinant(F);
  auto FinvT = Omega_h::transpose(Omega_h::invert(F));
  return J * sigma * FinvT;
}

TEST(composite_tet, first_pk) {
  auto X = get_reference_coords();
  auto x = get_current_coords();
  auto shape = lgr::CompTet::shape(X);
  Omega_h::Few<Omega_h::Matrix<3, 3>, 4> F_ips;
  for (int pt = 0;  pt < lgr::CompTet::points; ++pt) {
    auto BT = shape.basis_gradients[pt];
    auto B = Omega_h::transpose(BT);
    F_ips[pt] = x * B;
  }
  do_vol_avg_F(shape.weights, F_ips);
  auto first_pk_stress_gold = gold_first_pk_stress();
  for (int pt = 0; pt < lgr::CompTet::points; ++pt) {
    auto cauchy_stress = compute_sigma(F_ips[pt]);
    auto first_pk_stress = compute_intermediate_first_PK(
        F_ips[pt], cauchy_stress);
    auto first_pk_stress_sierra = to_sierra_full(first_pk_stress);
    EXPECT_TRUE(is_close(first_pk_stress_sierra, first_pk_stress_gold[pt]));
  }
}
