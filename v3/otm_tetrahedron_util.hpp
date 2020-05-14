#pragma once

#include <hpc_algorithm.hpp>
#include <hpc_array.hpp>
#include <hpc_array_vector.hpp>
#include <hpc_dimensional.hpp>
#include <hpc_execution.hpp>
#include <hpc_index.hpp>
#include <hpc_macros.hpp>
#include <hpc_range.hpp>
#include <hpc_range_sum.hpp>
#include <hpc_vector.hpp>
#include <hpc_vector3.hpp>
#include <lgr_mesh_indices.hpp>
#include <cassert>
#include <iostream>

namespace lgr
{

HPC_ALWAYS_INLINE HPC_HOST_DEVICE hpc::position<double>
tet_parametric_to_physical(hpc::array<hpc::position<double>, 4> const& xn,
    point_index const point, int const points_per_element)
{
  auto const ip = hpc::weaken(point) % points_per_element;
  hpc::position<double> xp(0.0, 0.0, 0.0);
  constexpr auto wa = 0.5854101966249685;
  constexpr auto wb = 0.1381966011250105;
  switch (points_per_element)
  {
  default:
    HPC_ERROR_EXIT("Invalid number of integration points");
    break;
  case 1:
    xp = 0.25 * (xn[0] + xn[1] + xn[2] + xn[3]);
    break;
  case 4:
    switch (ip)
    {
    default:
      HPC_ERROR_EXIT("Invalid integration point index");
      break;
    case 0:
      xp = wa * xn[0] + wb * (xn[1] + xn[2] + xn[3]);
      break;
    case 1:
      xp = wa * xn[1] + wb * (xn[0] + xn[2] + xn[3]);
      break;
    case 2:
      xp = wa * xn[2] + wb * (xn[0] + xn[1] + xn[3]);
      break;
    case 3:
      xp = wa * xn[3] + wb * (xn[0] + xn[1] + xn[2]);
      break;
    }
    break;
  }
  return xp;
}

struct tet_nodes_to_points
{
  tet_nodes_to_points(int const pts_per_elem) :
  points_per_element(pts_per_elem){}

  int const points_per_element{1};

  void operator()(hpc::counting_range<point_index> const points,
      hpc::device_range_sum<point_node_index, point_index> const &points_to_point_nodes,
      hpc::device_vector<node_index, point_node_index> const &point_nodes_to_nodes,
      hpc::device_array_vector<hpc::position<double>, node_index> const &x,
      hpc::device_array_vector<hpc::position<double>, point_index> &xp) const
  {
    auto pt_to_pt_nodes = points_to_point_nodes.cbegin();
    auto pt_nodes_to_nodes = point_nodes_to_nodes.cbegin();
    auto x_nodes = x.cbegin();
    auto x_points = xp.begin();
    auto const ppe = points_per_element;
    auto point_func = [=] HPC_DEVICE (point_index const point)
    {
      auto const point_nodes = pt_to_pt_nodes[point];
      hpc::array<hpc::position<double>, 4> x;
      assert(point_nodes.size() == 4);
      for (auto i = 0; i < 4; ++i)
      {
        auto const node = pt_nodes_to_nodes[point_nodes[i]];
        x[i] = x_nodes[node].load();
      }
      x_points[point] = tet_parametric_to_physical(x, point, ppe);
    };
    hpc::for_each(hpc::device_policy(), points, point_func);
  }
};

} // namespace lgr
