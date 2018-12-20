#include <lgr_ideal_gas.hpp>
#include "lgr_gtest.hpp"

TEST(ideal_gas, unit_inputs) {
  auto gamma = 1.4;
  auto rho = 1.0;
  auto e = 1.0;
  double p, c;
  lgr::ideal_gas_update(gamma, rho, e, p, c);
  EXPECT_TRUE(Omega_h::are_close(p, 0.4));
  EXPECT_TRUE(Omega_h::are_close(c, std::sqrt(1.4 * 0.4)));
}

LGR_END_TESTS
