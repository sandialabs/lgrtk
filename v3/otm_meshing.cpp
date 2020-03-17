#include <hpc_algorithm.hpp>
#include <hpc_atomic.hpp>
#include <hpc_execution.hpp>
#include <hpc_macros.hpp>
#include <hpc_numeric.hpp>
#include <hpc_range.hpp>
#include <hpc_vector.hpp>
#include <lgr_mesh_indices.hpp>
#include <otm_meshing_sort.hpp>
#include <lgr_state.hpp>
#include <otm_meshing.hpp>

namespace lgr {

void invert_otm_point_node_relations(lgr::state &s)
{
  auto points_to_point_nodes = s.points_to_point_nodes.cbegin();
  auto point_nodes_to_nodes = s.point_nodes_to_nodes.cbegin();
  hpc::device_vector<int, node_index> node_point_counts(s.nodes.size(), 0);
  auto counts = node_point_counts.begin();
  auto point_count_functor = [=] HPC_DEVICE(point_index const point) {
    auto point_node_range = points_to_point_nodes[point];
    for(auto point_node : point_node_range) {
      auto node = point_nodes_to_nodes[point_node];
      hpc::atomic_ref<int> count(counts[node]);
      count++;
    }
  };
  hpc::for_each(hpc::device_policy(), s.points, point_count_functor);
  auto total_node_points = hpc::reduce(hpc::device_policy(), node_point_counts, 0);

  s.node_points_to_points.resize(node_point_index(total_node_points));
  s.node_points_to_point_nodes.resize(node_point_index(total_node_points));
  s.nodes_to_node_points.assign_sizes(node_point_counts);
  hpc::fill(hpc::device_policy(), node_point_counts, 0);
  auto node_points_to_points = s.node_points_to_points.begin();
  auto nodes_to_node_points = s.nodes_to_node_points.cbegin();
  auto node_points_to_point_nodes = s.node_points_to_point_nodes.begin();
  auto node_point_fill_functor = [=] HPC_DEVICE(point_index const point)
  {
    auto const point_nodes = points_to_point_nodes[point];
    for (auto ni = 0; ni < point_nodes.size(); ++ni)
    {
      const auto point_node = point_nodes[ni];
      node_index const node = point_nodes_to_nodes[point_node];
      hpc::atomic_ref<int> count(counts[node]);
      int const offset = count++;
      auto const node_points_range = nodes_to_node_points[node];
      auto const node_point = node_points_range[point_node_index(offset)];
      node_points_to_points[node_point] = point;
      node_points_to_point_nodes[node_point] = ni;
    }
  };

  hpc::for_each(hpc::device_policy(), s.points, node_point_fill_functor);

  otm_sort_node_relations(s, s.nodes_to_node_points, s.node_points_to_points, s.node_points_to_point_nodes);
}

}
