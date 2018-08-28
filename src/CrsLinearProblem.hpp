//
//  CrsLinearProblem.hpp
//
//
//  Created by Roberts, Nathan V on 6/20/17.
//
//

#ifndef LGR_CRS_LINEAR_PROBLEM_HPP
#define LGR_CRS_LINEAR_PROBLEM_HPP

#include <CrsMatrix.hpp>
#include <Kokkos_Core.hpp>

namespace lgr {

template <class Ordinal>
class CrsLinearProblem {
 private:
  typedef Kokkos::View<Scalar *, MemSpace>          Vector;
  typedef Kokkos::View<Scalar **, Kokkos::LayoutRight, MemSpace>         MultiVector;
  typedef CrsMatrix<Ordinal, int> Matrix;

  Matrix _A;
  Vector _x, _b;  //left-hand side (solution), right-hand side
 public:
  CrsLinearProblem(const Matrix &Aa, Vector &ex, const Vector &be)
      : _A(Aa), _x(ex), _b(be) {}

  Matrix &A() { return _A; }
  Vector &b() { return _b; }
  Vector &x() { return _x; }

  virtual void initializeSolver() {}
  // concrete subclasses should know how to solve:
  virtual int solve() = 0;
};

}  // namespace lgr

#endif
