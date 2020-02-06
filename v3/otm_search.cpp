#include <hpc_algorithm.hpp>
#include <hpc_execution.hpp>
#include <hpc_functional.hpp>
#include <hpc_index.hpp>
#include <hpc_macros.hpp>
#include <hpc_numeric.hpp>
#include <hpc_range.hpp>
#include <hpc_vector.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Pair.hpp>
#include <lgr_mesh_indices.hpp>
#include <lgr_state.hpp>
#include <otm_arborx_search_impl.hpp>
#include <otm_meshing.hpp>
#include <otm_search.hpp>
#include <limits>

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

HPC_NOINLINE void do_otm_point_nearest_node_search(lgr::state &s, int max_support_nodes_per_point)
{
  auto search_nodes = arborx::create_arborx_nodes(s);
  auto search_points = arborx::create_arborx_points(s);
  auto queries = arborx::make_nearest_node_queries(search_points, max_support_nodes_per_point);

  arborx::device_int_view offsets;
  arborx::device_int_view indices;
  Kokkos::tie(offsets, indices) = arborx::do_search(search_nodes, queries);

  hpc::device_vector<int, point_index> counts(s.points.size());
  auto points_node_count = counts.begin();
  auto count_func = [=] HPC_DEVICE (lgr::point_index point)
  {
    auto point_begin = offsets(hpc::weaken(point));
    auto point_end = offsets(hpc::weaken(point) + 1);
    points_node_count[hpc::weaken(point)] = point_index(point_end - point_begin);
  };
  hpc::for_each(hpc::device_policy(), s.points, count_func);

  s.point_nodes.assign_sizes(counts);

  auto points_to_nodes_of_point = s.point_nodes.cbegin();
  auto points_to_supported_nodes = s.point_nodes_to_nodes.begin();
  auto fill_func = [=] HPC_DEVICE (lgr::point_index point)
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

namespace {

template <typename T, typename Range, typename Policy>
HPC_NOINLINE inline T reduce_min(Policy p, Range& r, T init)
{
  return hpc::transform_reduce(p, r, init, hpc::minimum<T>(), hpc::identity<T>());
}

}

void do_otm_iterative_point_support_search(lgr::state &s, int min_support_nodes_per_point)
{
  auto search_nodes = arborx::create_arborx_nodes(s);
  auto search_spheres = arborx::create_arborx_point_spheres(s);
  auto queries = arborx::make_intersect_sphere_queries(search_spheres);

  hpc::device_vector<int, point_index> counts(s.points.size());
  arborx::device_int_view offsets;
  arborx::device_int_view indices;
  int min_nodes_in_support_over_all_points = 0;
  const double inflation_factor = 1.2;
  while (min_nodes_in_support_over_all_points < min_support_nodes_per_point)
  {
    arborx::inflate_sphere_query_radii(queries, inflation_factor);

    Kokkos::tie(offsets, indices) = arborx::do_search(search_nodes, queries);

    auto points_node_count = counts.begin();
    auto count_func = [=] HPC_DEVICE (lgr::point_index point)
    {
      auto point_begin = offsets(hpc::weaken(point));
      auto point_end = offsets(hpc::weaken(point) + 1);
      points_node_count[hpc::weaken(point)] = point_index(point_end - point_begin);
    };
    hpc::for_each(hpc::device_policy(), s.points, count_func);

    min_nodes_in_support_over_all_points = reduce_min(hpc::device_policy(), counts,
        std::numeric_limits<int>::max());
  }

  s.point_nodes.assign_sizes(counts);

  auto new_points_to_support_size = hpc::reduce(hpc::device_policy(), counts, 0);
  s.point_nodes_to_nodes.resize(new_points_to_support_size);

  auto points_to_nodes_of_point = s.point_nodes.cbegin();
  auto points_to_supported_nodes = s.point_nodes_to_nodes.begin();
  auto fill_func = [=] HPC_DEVICE (lgr::point_index point)
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
