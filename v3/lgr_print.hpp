#pragma once

#include <iostream>
#include <hpc_index.hpp>
#include <hpc_dimensional.hpp>
#include <hpc_vector3.hpp>
#include <hpc_matrix3x3.hpp>
#include <hpc_symmetric3x3.hpp>

namespace lgr {

#ifdef HPC_STRONG_INDICES
template <class Tag, class Integral>
std::ostream& operator<<(std::ostream& stream, hpc::index<Tag, Integral> v) {
  stream << weaken(v);
  return stream;
}
#endif

#ifdef HPC_DIMENSIONAL_ANALYSIS
template <class T, class Dimension>
std::ostream& operator<<(std::ostream& stream, hpc::quantity<T, Dimension> v) {
  stream << weaken(v);
  return stream;
}
#endif

template <class Scalar>
std::ostream& operator<<(std::ostream& stream, hpc::vector3<Scalar> v) {
  stream << v(0) << " " << v(1) << " " << v(2);
  return stream;
}

template <class Scalar>
std::ostream& operator<<(std::ostream& stream, hpc::matrix3x3<Scalar> m) {
  stream << m(0, 0) << " " << m(0, 1) << " " << m(0, 2) << "\n";
  stream << m(1, 0) << " " << m(1, 1) << " " << m(1, 2) << "\n";
  stream << m(2, 0) << " " << m(2, 1) << " " << m(2, 2) << "\n";
  return stream;
}

template <class Scalar>
std::ostream& operator<<(std::ostream& stream, hpc::symmetric3x3<Scalar> m) {
  stream << m(0, 0) << " " << m(0, 1) << " " << m(0, 2) << "\n";
  stream << m(1, 0) << " " << m(1, 1) << " " << m(1, 2) << "\n";
  stream << m(2, 0) << " " << m(2, 1) << " " << m(2, 2) << "\n";
  return stream;
}

}
