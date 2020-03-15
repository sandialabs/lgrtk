#pragma once

#include <hpc_algorithm.hpp>
#include <hpc_array.hpp>
#include <hpc_execution.hpp>
#include <hpc_index.hpp>
#include <hpc_macros.hpp>
#include <hpc_range.hpp>
#include <hpc_vector.hpp>
#include <lgr_exodus.hpp>
#include <lgr_mesh_indices.hpp>
#include <otm_input.hpp>
#include <otm_state.hpp>
#include <otm_tet2meshless.hpp>

namespace lgr
{

HPC_ALWAYS_INLINE hpc::position<double>
tetrahedron_local_to_global_coord(const hpc::array<hpc::position<double>, 4>& node_coords,
    const hpc::position<double> &parametric_coord)
{
  return parametric_coord(0) * node_coords[0] + parametric_coord(1) * node_coords[1]
      + parametric_coord(2) * node_coords[2]
      + (1.0 - parametric_coord(0) - parametric_coord(1) - parametric_coord(2)) * node_coords[3];
}

struct tet_gauss_points_to_material_points
{
  tet_gauss_points_to_material_points(
      const hpc::host_vector<hpc::position<double>, point_node_index> &gauss_pts) :
      gauss_points(gauss_pts.size())
  {
    hpc::copy(gauss_pts, gauss_points);
  }

  void operator()(hpc::counting_range<point_index> const points,
      hpc::device_range_sum<point_node_index, point_index> const &points_to_point_nodes,
      hpc::device_vector<node_index, point_node_index> const &point_nodes_to_nodes,
      hpc::device_array_vector<hpc::position<double>, node_index> const &x,
      hpc::device_array_vector<hpc::position<double>, point_index> &xp)
  {
    auto pt_to_pt_nodes = points_to_point_nodes.cbegin();
    auto pt_nodes_to_nodes = point_nodes_to_nodes.cbegin();
    auto x_nodes = x.cbegin();
    auto x_points = xp.begin();
    auto gauss_pts = hpc::make_iterator_range(gauss_points.cbegin(), gauss_points.cend());
    auto point_func = [=] HPC_DEVICE(const point_index point)
    {
      auto const point_nodes = pt_to_pt_nodes[point];
      hpc::position<double> avg_coord(0., 0., 0.);
      hpc::array<hpc::position<double>, 4> x;
      assert(point_nodes.size() == 4);
      for (int i = 0; i < 4; ++i)
      {
        auto const node = pt_nodes_to_nodes[point_nodes[i]];
        x[i] = x_nodes[node].load();
      }
      for (auto gp : gauss_pts)
      {
        x_points[point] = tetrahedron_local_to_global_coord(x, gp);
      }
      avg_coord /= point_nodes.size();

      x_points[point] = avg_coord;
    };
    hpc::for_each(hpc::device_policy(), points, point_func);
  }

  hpc::device_vector<hpc::position<double>, point_node_index> gauss_points;
};

} // namespace lgr
