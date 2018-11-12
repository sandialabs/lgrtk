#ifndef LGR_GTEST_HPP
#define LGR_GTEST_HPP

#include <gtest/gtest.h>

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wglobal-constructors"
#endif

#ifdef __clang__
#define LGR_END_TESTS \
  _Pragma("clang diagnostic pop")
#else
#define LGR_END_TESTS
#endif

#endif
