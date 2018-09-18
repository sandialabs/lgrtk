#include <lgr_perfect_gas.hpp>
#include "lgr_gtest.hpp"

TEST(perfect_gas, zero_energy) {
  auto gamma = 1.4;
  auto rho = 1.0;
  auto e = 0.0;
  double p, c;
  lgr::perfect_gas_update(gamma, rho, e, p, c);
  EXPECT_TRUE(Omega_h::are_close(p, 0.0));
  EXPECT_TRUE(Omega_h::are_close(c, 0.0));
}

ALEXA_END_TESTS
