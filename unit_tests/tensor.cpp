#include <gtest/gtest.h>
#include <hpc_matrix3x3.hpp>

using Real = double;
using Tensor = hpc::matrix3x3<Real>;

int
main(int ac, char* av[])
{
  ::testing::GTEST_FLAG(print_time) = true;
  ::testing::InitGoogleTest(&ac, av);
  auto const retval = RUN_ALL_TESTS();
  return retval;
}

TEST(tensor, log)
{
  // Identity
  auto const eps = hpc::machine_epsilon<Real>();
  auto const I = Tensor::identity();
  auto const i = hpc::log(I);
  auto const error_I = hpc::norm(i) / hpc::norm(I);
  ASSERT_LE(error_I, eps);
  // 1/8 of a rotation
  auto const tau = 2.0 * std::acos(-1.0);
  auto const c = std::sqrt(2.0) / 2.0;
  auto const R = Tensor(c, -c, 0.0, c, c, 0.0, 0.0, 0.0, 1.0);
  auto const r = log(R);
  auto const error_R = std::abs(r(0,1) + tau / 8.0);
  ASSERT_LE(error_R, eps);
  auto const error_r = std::abs(r(0,1) + r(1,0));
  ASSERT_LE(error_r, eps);
}
