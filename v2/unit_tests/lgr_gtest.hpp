#ifndef ALEXA_GTEST_HPP
#define ALEXA_GTEST_HPP

#include <gtest/gtest.h>

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wglobal-constructors"
#endif

#ifdef __clang__
#define ALEXA_END_TESTS \
  _Pragma("clang diagnostic pop")
#else
#define ALEXA_END_TESTS
#endif

#endif
