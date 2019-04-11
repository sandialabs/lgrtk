#pragma once

#include <lgr_array.hpp>
#include <lgr_vector3.hpp>

namespace lgr {

inline double triangle_area(array<vector3<double>, 3> const x) {
  return 0.5 * (cross((x[1] - x[0]), (x[2] - x[0]))(2));
}

inline double tetrahedron_volume(array<vector3<double>, 4> const x) {
  return (1.0 / 6.0) * (cross((x[1] - x[0]), (x[2] - x[0])) * (x[3] - x[0]));
}

inline auto triangle_basis_gradients(
    array<vector3<double>, 3> const x,
    double const area) {
  vector3<double> edge_vectors[3];
  edge_vectors[0] = x[1] - x[0];
  edge_vectors[1] = x[2] - x[0];
  edge_vectors[2] = x[2] - x[1];
  constexpr vector3<double> z_axis(0.0, 0.0, 1.0);
  double const factor = 0.5 * (1.0 / area);
  array<vector3<double>, 3> grad_N;
  grad_N[0] = cross(z_axis, edge_vectors[2]) * factor;
  grad_N[1] = -cross(z_axis, edge_vectors[1]) * factor;
  grad_N[2] = cross(z_axis, edge_vectors[0]) * factor;
  return grad_N;
}

}
