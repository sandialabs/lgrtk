#pragma once

#include <iosfwd>
#include <lgr_vector3.hpp>
#include <lgr_matrix3x3.hpp>
#include <lgr_symmetric3x3.hpp>

namespace lgr {

std::ostream& operator<<(std::ostream&, vector3<double> v);
std::ostream& operator<<(std::ostream&, matrix3x3<double> v);
std::ostream& operator<<(std::ostream&, symmetric3x3<double> v);

}
