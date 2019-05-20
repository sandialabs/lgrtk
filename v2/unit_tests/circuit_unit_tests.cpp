#include <lgr_circuit.hpp>
#include <Omega_h_scalar.hpp>
#include "lgr_gtest.hpp"
#include <cmath>

TEST(circuit, RC) {
   EXPECT_TRUE(Omega_h::are_close(1.01, 1.0, 0.1, 0.0));
}

LGR_END_TESTS
