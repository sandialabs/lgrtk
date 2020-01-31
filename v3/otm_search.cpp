#include <hpc_algorithm.hpp>
#include <hpc_atomic.hpp>
#include <hpc_execution.hpp>
#include <hpc_index.hpp>
#include <hpc_macros.hpp>
#include <hpc_numeric.hpp>
#include <hpc_range.hpp>
#include <hpc_range_sum.hpp>
#include <hpc_vector.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Pair.hpp>
#include <lgr_mesh_indices.hpp>
#include <lgr_meshing_sort.hpp>
#include <lgr_state.hpp>
#include <otm_arborx_search_impl.hpp>
#include <cassert>

namespace lgr {
namespace search {

void initialize_otm_search()
{
  Kokkos::initialize();
}

void finalize_otm_search()
{
  Kokkos::finalize();
}

void invert_otm_point_node_relations(lgr::state &s)
{
  auto points_to_point_nodes = s.nodes_in_support.cbegin();
  auto point_nodes_to_nodes = s.points_to_supported_nodes.cbegin();
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

  s.nodes_to_influenced_points.resize(node_point_index(total_node_points));
  s.node_influenced_points_to_supporting_nodes.resize(node_point_index(total_node_points));
  s.points_in_influence.assign_sizes(node_point_counts);
  hpc::fill(hpc::device_policy(), node_point_counts, 0);
  auto node_points_to_points = s.nodes_to_influenced_points.begin();
  auto points_in_influence = s.points_in_influence.cbegin();
  auto node_points_to_supporting_nodes = s.node_influenced_points_to_supporting_nodes.begin();
  auto node_point_fill_functor = [=] HPC_DEVICE(point_index const point)
  {
    auto const point_nodes = points_to_point_nodes[point];
    for (auto ni = 0; ni < point_nodes.size(); ++ni)
    {
      const auto point_node = point_nodes[ni];
      node_index const node = point_nodes_to_nodes[point_node];
      hpc::atomic_ref<int> count(counts[node]);
      int const offset = count++;
      auto const node_points_range = points_in_influence[node];
      auto const node_point = node_points_range[node_in_support_index(offset)];
      node_points_to_points[node_point] = point;
      node_points_to_supporting_nodes[node_point] = ni;
    }
  };

  hpc::for_each(hpc::device_policy(), s.points, node_point_fill_functor);

  sort_node_relations(s, s.points_in_influence, s.nodes_to_influenced_points, s.node_influenced_points_to_supporting_nodes);
}

HPC_NOINLINE void do_otm_point_node_search(lgr::state &s, int max_support_nodes_per_point)
{
  auto search_nodes = arborx::create_arborx_nodes(s);
  auto search_points = arborx::create_arborx_points(s);
  auto queries = arborx::make_nearest_node_queries(search_points, max_support_nodes_per_point);

  arborx::device_int_view offsets;
  arborx::device_int_view indices;
  Kokkos::tie(offsets, indices) = arborx::do_search(search_nodes, queries);

  hpc::device_vector<int, point_index> counts(s.points.size());
  auto points_node_count = counts.begin();
  auto count_func = HPC_DEVICE [=](lgr::point_index point)
  {
    auto point_begin = offsets(hpc::weaken(point));
    auto point_end = offsets(hpc::weaken(point) + 1);
    points_node_count[hpc::weaken(point)] = point_index(point_end - point_begin);
  };
  hpc::for_each(hpc::device_policy(), s.points, count_func);

  s.nodes_in_support.assign_sizes(counts);

  auto points_to_nodes_of_point = s.nodes_in_support.cbegin();
  auto points_to_supported_nodes = s.points_to_supported_nodes.begin();
  auto fill_func = HPC_DEVICE [=](lgr::point_index point)
  {
    auto const point_nodes_range = points_to_nodes_of_point[point];
    for (auto point_node : point_nodes_range)
    {
      points_to_supported_nodes[point_node] = indices(hpc::weaken(point_node));
    }
  };
  hpc::for_each(hpc::device_policy(), s.points, fill_func);

  invert_otm_point_node_relations(s);
}

}
}
