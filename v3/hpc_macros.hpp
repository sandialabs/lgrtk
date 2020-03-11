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
#define HPC_TRACE_IMPL(msg, ...)                                      \
  do {                                                                \
  } while (0)
#else
#include <iostream>
#define HPC_TRACE_IMPL(msg, ...)                                      \
  do {                                                                \
    std::ostringstream omsg;                                          \
    omsg << "********** HPC_TRACE at ";                               \
    omsg << __FILE__ << " +" << __LINE__ << '\n' << msg << '\n';      \
    std::cerr << "********** HPC_TRACE at ";                          \
    std::cerr << __FILE__ << " +" << __LINE__ << "\n" << msg << '\n'; \
  } while (0)
#endif

#define HPC_TRACE(...) HPC_TRACE_IMPL(__VA_ARGS__, "")