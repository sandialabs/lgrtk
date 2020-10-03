#pragma once

#include <gtest/gtest.h>

#include <hpc_functional.hpp>
#include <hpc_transform_reduce.hpp>

#ifdef LGR_ENABLE_CUDA
#define DEVICE_TEST(a) [=] HPC_DEVICE(a, int& num_fails)
#define DEVICE_ASSERT_EQ(a, b) \
  if (a != b) {                \
    ++num_fails;               \
    return;                    \
  }
#define DEVICE_EXPECT_EQ(a, b) \
  if (a != b) {                \
    ++num_fails;               \
  }
#define DEVICE_ASSERT_NE(a, b) \
  if (a == b) {                \
    ++num_fails;               \
    return;                    \
  }
#define DEVICE_EXPECT_NE(a, b) \
  if (a == b) {                \
    ++num_fails;               \
  }
#define DEVICE_ASSERT_GT(a, b) \
  if (a <= b) {                \
    ++num_fails;               \
    return;                    \
  }
#define DEVICE_EXPECT_GT(a, b) \
  if (a <= b) {                \
    ++num_fails;               \
  }
#define DEVICE_ASSERT_LT(a, b) \
  if (a >= b) {                \
    ++num_fails;               \
    return;                    \
  }
#define DEVICE_EXPECT_LT(a, b) \
  if (a >= b) {                \
    ++num_fails;               \
  }
#define DEVICE_EXPECT_TRUE(a) \
  if (!a) {                   \
    ++num_fails;              \
  }
#define DEVICE_EXPECT_FALSE(a) \
  if (a) {                     \
    ++num_fails;               \
  }
#else
#define DEVICE_TEST(a) [=] HPC_DEVICE(a)
#define DEVICE_ASSERT_EQ(a, b) ASSERT_EQ(a, b)
#define DEVICE_EXPECT_EQ(a, b) EXPECT_EQ(a, b)
#define DEVICE_ASSERT_NE(a, b) ASSERT_NE(a, b)
#define DEVICE_EXPECT_NE(a, b) EXPECT_NE(a, b)
#define DEVICE_ASSERT_GT(a, b) ASSERT_GT(a, b)
#define DEVICE_EXPECT_GT(a, b) EXPECT_GT(a, b)
#define DEVICE_ASSERT_LT(a, b) ASSERT_LT(a, b)
#define DEVICE_EXPECT_LT(a, b) EXPECT_LT(a, b)
#define DEVICE_EXPECT_TRUE(a) EXPECT_TRUE(a)
#define DEVICE_EXPECT_FALSE(a) EXPECT_FALSE(a)
#endif

namespace unit {

namespace impl {

template <typename FuncType, typename Range>
class device_test
{
 public:
  device_test(const FuncType& func) : func(func)
  {
  }

  HPC_ALWAYS_INLINE HPC_DEVICE int
  operator()(typename Range::value_type i) const
  {
#ifdef LGR_ENABLE_CUDA
    int num_fails = 0;
    func(i, num_fails);
    return num_fails;
#else
    func(i);
    return 0;
#endif
  }

 private:
  const FuncType func;
};

}  // namespace impl

template <typename PolicyType, class Range, class FuncType>
HPC_NOINLINE void
test_for_each(PolicyType policy, Range const& range, const FuncType func)
{
  int  init(0);
  auto num_test_failures =
      hpc::transform_reduce(policy, range, init, hpc::plus<int>(), impl::device_test<FuncType, Range>(func));
  EXPECT_EQ(num_test_failures, 0);
}

}  // namespace unit
