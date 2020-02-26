#pragma once

#include <hpc_matrix3x3.hpp>
#include <iostream>

template <typename T>
std::ostream &
operator<<(std::ostream & os, hpc::matrix3x3<T> const & A)
{
  os << std::scientific << std::setprecision(15);
  for (auto i = 0; i < 3; ++i) {
    os << std::setw(24) << A(i,0);
    for (auto j = 1; j < 3; ++j) {
      os << "," << std::setw(24) << A(i,j);
    }
    os << std::endl;
  }
  return os;
}

