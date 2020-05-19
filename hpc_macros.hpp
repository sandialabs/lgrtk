#pragma once

#ifdef __CUDACC__
#define HPC_CUDA
#define HPC_HOST __host__
#define HPC_DEVICE __device__
#else
#define HPC_HOST
#define HPC_DEVICE
#endif

#define HPC_HOST_DEVICE HPC_HOST HPC_DEVICE

#if defined(DEBUG)
#define HPC_NOINLINE __attribute__((noinline))
#else
#define HPC_NOINLINE
#endif
#define HPC_ALWAYS_INLINE __attribute__((always_inline)) inline

// Macros for debugging
#ifdef __CUDACC__
#else
#include <cmath>
#include <pmmintrin.h>
#include <xmmintrin.h>
#include <iostream>
#include <sstream>
#endif

#ifdef __CUDACC__
#define HPC_TRACE_IMPL(msg)                                            \
  do {                                                                 \
      printf("%s ********** HPC_TRACE at ");                           \
      printf("%s +%d\n%s\n", __FILE__, __LINE__, msg);                 \
  } while (0)
#else
#define HPC_TRACE_IMPL(msg)                                            \
  do {                                                                 \
    std::cout << "********** HPC_TRACE at ";                           \
    std::cout << __FILE__ << " +" << __LINE__ << "\n" << msg << '\n';  \
  } while (0)
#endif

#ifdef __CUDACC__
#define HPC_DUMP_IMPL(msg)                                             \
  do {                                                                 \
      printf("%s", msg);                                               \
  } while (0)
#else
#define HPC_DUMP_IMPL(msg)                                             \
  do {                                                                 \
    std::cout << msg;                                                  \
  } while (0)
#endif

#ifdef __CUDACC__
#define HPC_ERROR_EXIT_IMPL(msg)                                       \
  do {                                                                 \
    printf("********** HPC_ERROR at ");                                \
    printf("%s:%d\n  %s", __FILE__, __LINE__, msg);                    \
    assert(0);                                                         \
  } while (0)
#else
#define HPC_ERROR_EXIT_IMPL(msg)                                       \
  do {                                                                 \
    std::cout << "********** HPC ERROR at ";                           \
    std::cout << __FILE__ << " +" << __LINE__ << "\n" << msg << '\n';  \
    exit(1);                                                           \
  } while (0)
#endif

#ifdef __CUDACC__
#define HPC_TRAP_FPE_IMPL(...)                                         \
  do {                                                                 \
  } while (0)
#else
#define HPC_TRAP_FPE_IMPL(...)                                         \
  do {                                                                 \
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);                          \
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);                  \
  _MM_SET_EXCEPTION_MASK(_MM_GET_EXCEPTION_MASK() &                    \
      ~(_MM_MASK_INVALID | _MM_MASK_DIV_ZERO | _MM_MASK_OVERFLOW));    \
  } while (0)
#endif

#ifdef __CUDACC__
#define HPC_ASSERT_IMPL(cond, msg)                                     \
  do {                                                                 \
    if (!(cond)) {                                                     \
      printf("%s ********** HPC_ASSERT failed at ", #cond);            \
      printf("%s +%d\n%s\n", __FILE__, __LINE__, msg);                 \
      assert(0);                                                       \
    }                                                                  \
  } while (0)
#else
#define HPC_ASSERT_IMPL(cond, msg, ...)                                \
  do {                                                                 \
    if (!(cond)) {                                                     \
      std::cout << #cond " ********** HPC_ASSERT failed at ";          \
      std::cout << __FILE__ << " +" << __LINE__ << "\n" << msg << '\n';\
      abort();                                                         \
    }                                                                  \
  } while (0)
#endif

#define HPC_TRACE(...) HPC_TRACE_IMPL(__VA_ARGS__)
#define HPC_DUMP(...) HPC_DUMP_IMPL(__VA_ARGS__)
#define HPC_ERROR_EXIT(...) HPC_ERROR_EXIT_IMPL(__VA_ARGS__)
#define HPC_TRAP_FPE(...) HPC_TRAP_FPE_IMPL(__VA_ARGS__)
#define HPC_ASSERT(...) HPC_ASSERT_IMPL(__VA_ARGS__, "")
