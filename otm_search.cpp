#include <hpc_algorithm.hpp>
#include <hpc_array_vector.hpp>
#include <hpc_execution.hpp>
#include <hpc_index.hpp>
#include <hpc_macros.hpp>
#include <hpc_range.hpp>
#include <hpc_range_sum.hpp>
#include <hpc_vector.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Pair.hpp>
#include <lgr_mesh_indices.hpp>
#include <lgr_state.hpp>
#include <otm_arborx_search_impl.hpp>

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

HPC_NOINLINE void do_otm_point_node_search(lgr::state &s,
    hpc::device_range_sum<point_node_index, point_index> &points_to_point_nodes)
{
  auto search_nodes = arborx::create_arborx_nodes(s);
  auto search_points = arborx::create_arborx_points(s);
  const int num_nodes_per_point_to_find = 4;
  auto queries = arborx::make_nearest_node_queries(search_points,
      num_nodes_per_point_to_find);

  arborx::device_int_view offsets;
  arborx::device_int_view indices;
  Kokkos::tie(offsets, indices) = arborx::do_search(search_nodes, queries);

  hpc::device_vector<int, point_index> counts(s.points.size());
  auto points_node_count = counts.begin();
  auto count_func =
      HPC_DEVICE [=](
          lgr::point_index point)
          {
            auto point_begin = offsets(hpc::weaken(point));
            auto point_end = offsets(hpc::weaken(point) + 1);
            points_node_count[hpc::weaken(point)] = point_index(point_end - point_begin);
          };
  hpc::for_each(hpc::device_policy(), s.points, count_func);

  points_to_point_nodes.assign_sizes(counts);

  auto points_to_nodes_of_point = points_to_point_nodes.cbegin();
  auto points_to_supported_nodes = s.points_to_supported_nodes.begin();
  auto fill_func =
      HPC_DEVICE [=](
          lgr::point_index point)
          {
            auto const point_nodes_range = points_to_nodes_of_point[point];
            for (auto point_node : point_nodes_range)
            {
              points_to_supported_nodes[point_node] = indices(hpc::weaken(point_node));
            }
          };
}

}
}
