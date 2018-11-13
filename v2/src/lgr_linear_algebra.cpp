#include <lgr_linear_algebra.hpp>
#include <Omega_h_profile.hpp>
#include <lgr_for.hpp>
#include <Omega_h_array_ops.hpp>

namespace lgr {

void matvec(GlobalMatrix mat, GlobalVector vec, GlobalVector result) {
  OMEGA_H_TIME_FUNCTION;
  auto const n = result.size();
  auto f = OMEGA_H_LAMBDA(int const row) {
    double value = 0.0;
    auto const begin = mat.rows_to_columns.a2ab[row];
    auto const end = mat.rows_to_columns.a2ab[row + 1];
    for (auto row_col = begin; row_col < end; ++row_col) {
      auto const col = mat.rows_to_columns.ab2b[row_col];
      value += mat.entries[row_col] * vec[col];
    }
    result[row] = value;
  };
  parallel_for(n, std::move(f));
}

double dot(GlobalVector a, GlobalVector b) {
  OMEGA_H_TIME_FUNCTION;
  auto const tmp = multiply_each(read(a), read(b), "dot tmp");
  return repro_sum(read(tmp));
}

void axpy(double a, GlobalVector x, GlobalVector y, GlobalVector result) {
  OMEGA_H_TIME_FUNCTION;
  auto f = OMEGA_H_LAMBDA(int const i) {
    result[i] = a * x[i] + y[i];
  };
  parallel_for(result.size(), std::move(f));
}

int conjugate_gradient(
    GlobalMatrix A,
    GlobalVector b,
    GlobalVector x,
    double max_residual_magnitude) {
  OMEGA_H_TIME_FUNCTION;
  auto const n = x.size();
  GlobalVector r(n, "CG/r");
  matvec(A, x, r); // r = A * x
  axpy(-1.0, r, b, r); // r = -r + b, r = b - A * x
  GlobalVector p(n, "CG/p");
  Omega_h::copy_into(read(r), p); // p = r
  auto rsold = dot(r, r);
  if (std::sqrt(rsold) < max_residual_magnitude) {
    return 0;
  }
  GlobalVector Ap(n, "CG/Ap");
  for (int i = 0; i < b.size(); ++i) {
    matvec(A, p, Ap);
    auto const alpha = rsold / dot(p, Ap);
    axpy(alpha, p, x, x); // x = x + alpha * p
    axpy(-alpha, Ap, r, r); // r = r - alpha * Ap
    auto const rsnew = dot(r, r);
    if (std::sqrt(rsnew) < max_residual_magnitude) {
      return i + 1;
    }
    auto const beta = rsnew / rsold;
    axpy(beta, p, r, p); // p = r + (rsnew / rsold) * p
    rsold = rsnew;
  }
  return b.size() + 1;
}

}
