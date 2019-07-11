#include <iostream>

#include <lgr_print.hpp>

namespace lgr {

std::ostream& operator<<(std::ostream& stream, hpc::vector3<double> v) {
  stream << v(0) << " " << v(1) << " " << v(2);
  return stream;
}

std::ostream& operator<<(std::ostream& stream, hpc::matrix3x3<double> m) {
  stream << m(0, 0) << " " << m(0, 1) << " " << m(0, 2) << "\n";
  stream << m(1, 0) << " " << m(1, 1) << " " << m(1, 2) << "\n";
  stream << m(2, 0) << " " << m(2, 1) << " " << m(2, 2) << "\n";
  return stream;
}

std::ostream& operator<<(std::ostream& stream, hpc::symmetric3x3<double> m) {
  stream << m(0, 0) << " " << m(0, 1) << " " << m(0, 2) << "\n";
  stream << m(1, 0) << " " << m(1, 1) << " " << m(1, 2) << "\n";
  stream << m(2, 0) << " " << m(2, 1) << " " << m(2, 2) << "\n";
  return stream;
}

}
