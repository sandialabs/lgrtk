#pragma once

#include <cfloat>
#include <climits>
#include <cstdint>
#include <hpc_macros.hpp>
#include <limits>

namespace hpc {

template <class T>
class numeric_limits;

template <>
class numeric_limits<char>
{
 public:
  static constexpr bool is_integer = true;
  static constexpr bool is_signed  = std::numeric_limits<char>::is_signed;
  HPC_HOST_DEVICE static constexpr char
  min() noexcept
  {
    return CHAR_MIN;
  }
  HPC_HOST_DEVICE static constexpr char
  lowest() noexcept
  {
    return CHAR_MIN;
  }
  HPC_HOST_DEVICE static constexpr char
  max() noexcept
  {
    return CHAR_MAX;
  }
};

template <>
class numeric_limits<signed char>
{
 public:
  static constexpr bool is_integer = true;
  static constexpr bool is_signed  = true;
  HPC_HOST_DEVICE static constexpr signed char
  min() noexcept
  {
    return SCHAR_MIN;
  }
  HPC_HOST_DEVICE static constexpr signed char
  lowest() noexcept
  {
    return SCHAR_MIN;
  }
  HPC_HOST_DEVICE static constexpr signed char
  max() noexcept
  {
    return SCHAR_MAX;
  }
};

template <>
class numeric_limits<unsigned char>
{
 public:
  static constexpr bool is_integer = true;
  static constexpr bool is_signed  = false;
  HPC_HOST_DEVICE static constexpr unsigned char
  min() noexcept
  {
    return 0;
  }
  HPC_HOST_DEVICE static constexpr unsigned char
  lowest() noexcept
  {
    return 0;
  }
  HPC_HOST_DEVICE static constexpr unsigned char
  max() noexcept
  {
    return UCHAR_MAX;
  }
};

template <>
class numeric_limits<short>
{
 public:
  static constexpr bool is_integer = true;
  static constexpr bool is_signed  = true;
  HPC_HOST_DEVICE static constexpr short
  min() noexcept
  {
    return SHRT_MIN;
  }
  HPC_HOST_DEVICE static constexpr short
  lowest() noexcept
  {
    return SHRT_MIN;
  }
  HPC_HOST_DEVICE static constexpr short
  max() noexcept
  {
    return SHRT_MAX;
  }
};

template <>
class numeric_limits<unsigned short>
{
 public:
  static constexpr bool is_integer = true;
  static constexpr bool is_signed  = false;
  HPC_HOST_DEVICE static constexpr unsigned short
  min() noexcept
  {
    return 0;
  }
  HPC_HOST_DEVICE static constexpr unsigned short
  lowest() noexcept
  {
    return 0;
  }
  HPC_HOST_DEVICE static constexpr unsigned short
  max() noexcept
  {
    return USHRT_MAX;
  }
};

template <>
class numeric_limits<int>
{
 public:
  static constexpr bool is_integer = true;
  static constexpr bool is_signed  = true;
  HPC_HOST_DEVICE static constexpr int
  min() noexcept
  {
    return INT_MIN;
  }
  HPC_HOST_DEVICE static constexpr int
  lowest() noexcept
  {
    return INT_MIN;
  }
  HPC_HOST_DEVICE static constexpr int
  max() noexcept
  {
    return INT_MAX;
  }
};

template <>
class numeric_limits<unsigned int>
{
 public:
  static constexpr bool is_integer = true;
  static constexpr bool is_signed  = false;
  HPC_HOST_DEVICE static constexpr unsigned int
  min() noexcept
  {
    return 0;
  }
  HPC_HOST_DEVICE static constexpr unsigned int
  lowest() noexcept
  {
    return 0;
  }
  HPC_HOST_DEVICE static constexpr unsigned int
  max() noexcept
  {
    return UINT_MAX;
  }
};

template <>
class numeric_limits<long>
{
 public:
  static constexpr bool is_integer = true;
  static constexpr bool is_signed  = true;
  HPC_HOST_DEVICE static constexpr long
  min() noexcept
  {
    return LONG_MIN;
  }
  HPC_HOST_DEVICE static constexpr long
  lowest() noexcept
  {
    return LONG_MIN;
  }
  HPC_HOST_DEVICE static constexpr long
  max() noexcept
  {
    return LONG_MAX;
  }
};

template <>
class numeric_limits<unsigned long>
{
 public:
  static constexpr bool is_integer = true;
  static constexpr bool is_signed  = false;
  HPC_HOST_DEVICE static constexpr unsigned long
  min() noexcept
  {
    return 0;
  }
  HPC_HOST_DEVICE static constexpr unsigned long
  lowest() noexcept
  {
    return 0;
  }
  HPC_HOST_DEVICE static constexpr unsigned long
  max() noexcept
  {
    return ULONG_MAX;
  }
};

template <>
class numeric_limits<long long>
{
 public:
  static constexpr bool is_integer = true;
  static constexpr bool is_signed  = true;
  HPC_HOST_DEVICE static constexpr long long
  min() noexcept
  {
    return LLONG_MIN;
  }
  HPC_HOST_DEVICE static constexpr long long
  lowest() noexcept
  {
    return LLONG_MIN;
  }
  HPC_HOST_DEVICE static constexpr long long
  max() noexcept
  {
    return LLONG_MAX;
  }
};

template <>
class numeric_limits<unsigned long long>
{
 public:
  static constexpr bool is_integer = true;
  static constexpr bool is_signed  = false;
  HPC_HOST_DEVICE static constexpr unsigned long long
  min() noexcept
  {
    return 0;
  }
  HPC_HOST_DEVICE static constexpr unsigned long long
  lowest() noexcept
  {
    return 0;
  }
  HPC_HOST_DEVICE static constexpr unsigned long long
  max() noexcept
  {
    return ULLONG_MAX;
  }
};

template <>
class numeric_limits<float>
{
 public:
  static constexpr bool is_integer = false;
  static constexpr bool is_signed  = true;
  HPC_HOST_DEVICE static constexpr float
  min() noexcept
  {
    return FLT_MIN;
  }
  HPC_HOST_DEVICE static constexpr float
  lowest() noexcept
  {
    return -FLT_MAX;
  }
  HPC_HOST_DEVICE static constexpr float
  max() noexcept
  {
    return FLT_MAX;
  }
  HPC_HOST_DEVICE static constexpr float
  infinity() noexcept
  {
    return HUGE_VALF;
  }
};

template <>
class numeric_limits<double>
{
 public:
  static constexpr bool is_integer = false;
  static constexpr bool is_signed  = true;
  HPC_HOST_DEVICE static constexpr double
  min() noexcept
  {
    return DBL_MIN;
  }
  HPC_HOST_DEVICE static constexpr double
  lowest() noexcept
  {
    return -DBL_MAX;
  }
  HPC_HOST_DEVICE static constexpr double
  max() noexcept
  {
    return DBL_MAX;
  }
  HPC_HOST_DEVICE static constexpr double
  infinity() noexcept
  {
    return HUGE_VAL;
  }
};

}  // namespace hpc
