#include <lgr_linear_algebra.hpp>
#include <Omega_h_profile.hpp>
#include <lgr_for.hpp>
#include <Omega_h_array_ops.hpp>

//DEBUG!
#include <iostream>

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
  std::cout << "rsold = " << rsold << '\n';
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
    std::cout << "rsnew[" << i <<"] = " << rsnew << '\n';
    if (std::sqrt(rsnew) < max_residual_magnitude) {
      return i + 1;
    }
    auto const beta = rsnew / rsold;
    axpy(beta, p, r, p); // p = r + (rsnew / rsold) * p
    rsold = rsnew;
  }
  return b.size() + 1;
}

void set_boundary_conditions(
    GlobalMatrix A,
    GlobalVector x,
    GlobalVector b,
    Omega_h::LOs rows_to_bc_rows) {
  auto functor = OMEGA_H_LAMBDA(int row) {
    auto const begin = A.rows_to_columns.a2ab[row];
    auto const end = A.rows_to_columns.a2ab[row + 1];
    if (rows_to_bc_rows[row] == -1) {
      auto row_b = b[row];
      for (auto row_col = begin; row_col < end; ++row_col) {
        auto const col = A.rows_to_columns.ab2b[row_col];
        if (rows_to_bc_rows[col] != -1) {
          row_b -= A.entries[row_col] * x[col];
          A.entries[row_col] = 0.0;
        }
      }
      b[row] = row_b;
    } else {
      for (auto row_col = begin; row_col < end; ++row_col) {
        auto const col = A.rows_to_columns.ab2b[row_col];
        if (col == row) {
          b[row] = A.entries[row_col] * x[row];
        } else {
          A.entries[row_col] = 0.0;
        }
      }
    }
  };
  parallel_for(x.size(), std::move(functor));
}

MediumMatrix::MediumMatrix(int const size_in) {
  size = size_in;
  entries.resize(std::size_t(size_in * size_in), 0.0);
}

MediumVector::MediumVector(int const size_in) {
  entries.resize(std::size_t(size_in), 0.0);
}

void gaussian_elimination(MediumMatrix& A, MediumVector& b) {
//if ((1)) return;
  int h = 0; /* Initialization of the pivot row */
  int k = 0; /* Initialization of the pivot column */
  auto const m = A.size;
  while (h < m && k < m) {
    int i_max = -1;
    double i_max_value = 0.0;
    /* Find the k-th pivot: */
    for (int i = h; i < m; ++i) {
      double const i_value = std::abs(A(i, k));
      if (i_max == -1 || i_value > i_max_value) {
        i_max_value = i_value;
        i_max = i;
      }
    }
    if (A(i_max, k) == 0.0) {
      /* No pivot in this column, pass to next column */
      ++k;
    } else {
      /* swap rows */
      for (int j = 0; j < m; ++j) {
        std::swap(A(h, j), A(i_max, j));
      }
      std::swap(b(h), b(i_max));
      /* Do for all rows below pivot: */
      for (int i = h + 1; i < m; ++i) {
        auto const f = A(i, k) / A(h, k);
        /* Fill with zeros the lower part of pivot column: */
        A(i, k) = 0.0;
        /* Do for all remaining elements in current row: */
        for (int j = k + 1; j < m; ++j) {
          A(i, j) -= A(h, j) * f;
        }
        b(i) -= b(h) * f;
      }
      /* Increase pivot row and column */
      ++h;
      ++k;
    }
  }
}

void back_substitution(MediumMatrix const& A, MediumVector const& b, MediumVector& x) {
  auto const n = A.size;
  x = MediumVector(n);
  for (int ri = 0; ri < n; ++ri) {
    int const i = n - ri - 1;
    double xi = b(i);
    for (int j = i + 1; j < n; ++j) {
      xi -= A(i, j) * x(j);
    }
    xi /= A(i, i);
    x(i) = xi;
  }
}

}
