#pragma once

#include <hpc_macros.hpp>

namespace hpc {

class local_policy {};
class serial_policy {};
class cuda_policy {};

using host_policy = serial_policy;
#ifdef HPC_CUDA
using device_policy = cuda_policy;
#else
using device_policy = serial_policy;
#endif

}
