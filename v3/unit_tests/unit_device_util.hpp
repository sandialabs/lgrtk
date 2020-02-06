#include <gtest/gtest.h>

#include <hpc_functional.hpp>
#include <hpc_transform_reduce.hpp>

#define DEVICE_TEST(a) [=] HPC_DEVICE(a, int& num_fails)
#define DEVICE_ASSERT_EQ(a, b) if (a != b) { ++num_fails; return; }
#define DEVICE_EXPECT_EQ(a, b) if (a != b) { ++num_fails; }
#define DEVICE_EXPECT_TRUE(a) if (!a) { ++num_fails; }
#define DEVICE_EXPECT_FALSE(a) if (a) { ++num_fails; }

namespace unit {

namespace impl {

template<typename FuncType, typename Range>
class device_test
{
public:
  device_test(const FuncType &func) :
      func(func)
  {
  }

  HPC_ALWAYS_INLINE HPC_DEVICE int operator()(typename Range::value_type i) const
  {
    int num_fails = 0;
    func(i, num_fails);
    return num_fails;
  }

private:
  const FuncType &func;
};

}

template<typename PolicyType, class Range, class FuncType>
HPC_NOINLINE void test_for_each(PolicyType policy, Range const &range, const FuncType &func)
{
  int init(0);
  auto num_test_failures = hpc::transform_reduce(policy, range, init, hpc::plus<int>(),
      impl::device_test<FuncType, Range>(func));
  EXPECT_EQ(num_test_failures, 0);
}

}

