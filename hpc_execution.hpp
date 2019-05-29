#pragma once

#include <hpc_macros.hpp>

namespace hpc {

class local_policy {};
class serial_policy {};

using host_policy = serial_policy;
using device_policy = serial_policy;
using host_to_device_policy = serial_policy;

}
