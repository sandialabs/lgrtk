#include <gtest/gtest.h>

int
main(int argc, char* argv[])
{
  ::testing::GTEST_FLAG(print_time) = true;
  ::testing::InitGoogleTest(&argc, argv);
  auto const retval = RUN_ALL_TESTS();
  return retval;
}
