#pragma once

#include <iomanip>
#include <iostream>

#include <hpc_array_vector.hpp>
#include <hpc_matrix3x3.hpp>
#include <hpc_quaternion.hpp>
#include <hpc_vector3.hpp>

template <typename T>
std::ostream &
operator<<(std::ostream & os, hpc::vector3<T> const & v)
{
  os << std::scientific << std::setprecision(15);
  os << std::setw(24) << v(0) << "," << std::setw(24) << v(1) << "," << std::setw(24) << v(2);
  return os;
}

template <typename T>
std::ostream &
operator<<(std::ostream & os, hpc::quaternion<T> const & q)
{
  os << std::scientific << std::setprecision(15);
  os << std::setw(24) << q(0) << "," << std::setw(24) << q(1) << "," << std::setw(24) << q(2) << "," << std::setw(24) << q(3);
  return os;
}

template <typename T>
std::ostream &
operator<<(std::ostream & os, hpc::matrix3x3<T> const & A)
{
  os << std::scientific << std::setprecision(15);
  os << std::setw(24) << A(0,0) << "," << std::setw(24) << A(0,1) << "," << std::setw(24) << A(0,2) << std::endl;
  os << std::setw(24) << A(1,0) << "," << std::setw(24) << A(1,1) << "," << std::setw(24) << A(1,2) << std::endl;
  os << std::setw(24) << A(2,0) << "," << std::setw(24) << A(2,1) << "," << std::setw(24) << A(2,2);
  return os;
}
