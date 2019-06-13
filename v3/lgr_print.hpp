#pragma once

#include <iosfwd>
#include <hpc_vector3.hpp>
#include <hpc_matrix3x3.hpp>
#include <hpc_symmetric3x3.hpp>

namespace lgr {

std::ostream& operator<<(std::ostream&, hpc::vector3<double> v);
std::ostream& operator<<(std::ostream&, hpc::matrix3x3<double> v);
std::ostream& operator<<(std::ostream&, hpc::symmetric3x3<double> v);

}
