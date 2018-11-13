#include <lgr_mie_gruneisen.hpp>
#include "lgr_gtest.hpp"

TEST(mie_gruneisen, zero_energy) {
  double rho0 = 2700;
  double gamma0 = 1.5;
  double c0 = 5400;
  double s1 = 1.4;
  auto rho = 2700;
  auto e = 0.0;
  double p, c;
  lgr::mie_gruneisen_update(rho0, gamma0, c0, s1, rho, e, p, c);
  EXPECT_TRUE(Omega_h::are_close(p, 0.0));
  EXPECT_TRUE(Omega_h::are_close(c, c0));
}

LGR_END_TESTS
