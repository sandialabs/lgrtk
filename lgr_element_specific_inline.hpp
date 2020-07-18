#pragma once

#include <hpc_array.hpp>
#include <hpc_dimensional.hpp>
#include <hpc_vector3.hpp>

namespace lgr {

HPC_HOST_DEVICE inline hpc::area<double>
triangle_area(hpc::array<hpc::position<double>, 3> const x) noexcept
{
  return 0.5 * (cross((x[1] - x[0]), (x[2] - x[0]))(2));
}

HPC_HOST_DEVICE inline hpc::volume<double>
tetrahedron_volume(hpc::array<hpc::position<double>, 4> const x)
{
  return (1.0 / 6.0) * (cross((x[1] - x[0]), (x[2] - x[0])) * (x[3] - x[0]));
}

HPC_HOST_DEVICE inline auto
triangle_basis_gradients(
    hpc::array<hpc::position<double>, 3> const x,
    hpc::area<double> const                    area) noexcept
{
  hpc::displacement<double> edge_vectors[3];
  edge_vectors[0] = x[1] - x[0];
  edge_vectors[1] = x[2] - x[0];
  edge_vectors[2] = x[2] - x[1];
  constexpr hpc::vector3<double>             z_axis(0.0, 0.0, 1.0);
  auto const                                 factor = 0.5 * (1.0 / area);
  hpc::array<hpc::basis_gradient<double>, 3> grad_N;
  grad_N[0] = cross(z_axis, edge_vectors[2]) * factor;
  grad_N[1] = -cross(z_axis, edge_vectors[1]) * factor;
  grad_N[2] = cross(z_axis, edge_vectors[0]) * factor;
  return grad_N;
}

HPC_HOST_DEVICE inline auto
tetrahedron_basis_gradients(
    hpc::array<hpc::position<double>, 4> const x,
    hpc::volume<double> const                  volume) noexcept
{
  hpc::displacement<double> ev[5];
  ev[0]             = x[1] - x[0];
  ev[1]             = x[2] - x[0];
  ev[2]             = x[3] - x[0];
  ev[3]             = x[2] - x[1];
  ev[4]             = x[3] - x[1];
  auto const factor = (1.0 / 6.0) * (1.0 / volume);
  hpc::array<hpc::basis_gradient<double>, 4> grad_N;
  grad_N[0] = cross(ev[4], ev[3]) * factor;
  grad_N[1] = cross(ev[1], ev[2]) * factor;
  grad_N[2] = cross(ev[2], ev[0]) * factor;
  grad_N[3] = cross(ev[0], ev[1]) * factor;
  return grad_N;
}

}  // namespace lgr
