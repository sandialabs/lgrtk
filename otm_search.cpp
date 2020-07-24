#include <hpc_execution.hpp>
#include <hpc_functional.hpp>
#include <hpc_macros.hpp>
#include <hpc_range.hpp>
#include <hpc_range_sum.hpp>
#include <hpc_vector.hpp>
#include <lgr_mesh_indices.hpp>
#include <lgr_state.hpp>
#include <limits>
#include <otm_meshing.hpp>
#include <otm_search.hpp>
#include <otm_search_util.hpp>

using namespace lgr::search_util;

#ifdef LGR_ENABLE_SEARCH
#include <otm_arborx_search_impl.hpp>
#endif

namespace lgr {
namespace search {

#ifdef LGR_ENABLE_SEARCH

void
initialize_otm_search()
{
  arborx::initialize();
}

void
finalize_otm_search()
{
  arborx::finalize();
}

namespace {

template <typename QueryViewType, typename IndexType>
HPC_NOINLINE void
do_search_and_fill_counts(
    const arborx::device_point_view&      search_points,
    const QueryViewType&                  queries,
    const hpc::counting_range<IndexType>& search_indices,
    arborx::device_int_view&              offsets,
    arborx::device_int_view&              indices,
    hpc::device_vector<int, IndexType>&   search_counts)
{
  arborx::do_search(search_points, queries, indices, offsets);

  auto counts     = search_counts.begin();
  auto count_func = [=] HPC_DEVICE(IndexType point) {
    auto point_begin = offsets(hpc::weaken(point));
    auto point_end   = offsets(hpc::weaken(point) + 1);
    counts[point]    = int(point_end - point_begin);
  };
  hpc::for_each(hpc::device_policy(), search_indices, count_func);
}

template <typename Index1, typename Index2, typename Index1to2Ordinal>
HPC_NOINLINE void
size_and_fill_lgr_data_structures(
    const hpc::counting_range<Index1>&               search_indices,
    const arborx::device_int_view&                   indices,
    const hpc::device_vector<int, Index1>&           search_counts,
    hpc::device_range_sum<Index1to2Ordinal, Index1>& search_result_ranges,
    hpc::device_vector<Index2, Index1to2Ordinal>&    search_results)
{
  search_result_ranges.assign_sizes(search_counts);
  auto new_results_size = hpc::reduce(hpc::device_policy(), search_counts, 0);
  search_results.resize(new_results_size);

  auto points_results_ranges = search_result_ranges.cbegin();
  auto points_to_results     = search_results.begin();
  auto fill_func             = [=] HPC_DEVICE(Index1 point) {
    auto const point_results_range = points_results_ranges[point];
    for (auto result : point_results_range) { points_to_results[result] = Index2(indices(hpc::weaken(result))); }
  };
  hpc::for_each(hpc::device_policy(), search_indices, fill_func);
}

template <typename IndexType, typename IndexOrdinalType>
HPC_NOINLINE void
size_and_fill_lgr_data_structures_from_symmetric_search(
    const hpc::counting_range<IndexType>&               search_indices,
    const arborx::device_int_view&                      indices,
    const arborx::device_int_view&                      offsets,
    hpc::device_vector<int, IndexType>&                 search_counts,
    hpc::device_range_sum<IndexOrdinalType, IndexType>& search_result_ranges,
    hpc::device_vector<IndexType, IndexOrdinalType>&    search_results)
{
  // this version filters search indices from the results where index == result,
  // which implies that a node or point found itself in the search.

  auto counts     = search_counts.begin();
  auto count_func = [=] HPC_DEVICE(IndexType point) { counts[point] -= 1; };
  hpc::for_each(hpc::device_policy(), search_indices, count_func);

  search_result_ranges.assign_sizes(search_counts);
  auto new_results_size = hpc::reduce(hpc::device_policy(), search_counts, 0);
  search_results.resize(new_results_size);

  auto points_results_ranges = search_result_ranges.cbegin();
  auto points_to_results     = search_results.begin();
  auto fill_func             = [=] HPC_DEVICE(IndexType point) {
    auto const point_results_range   = points_results_ranges[point];
    auto       non_self_result_index = offsets(hpc::weaken(point));
    for (auto result : point_results_range) {
      if (indices(non_self_result_index) == point) ++non_self_result_index;
      points_to_results[result] = IndexType(indices(non_self_result_index));
      ++non_self_result_index;
    }
  };
  hpc::for_each(hpc::device_policy(), search_indices, fill_func);
}

template <typename QueryViewType, typename Index1, typename Index2, typename Index1to2Ordinal>
HPC_NOINLINE void
do_search_and_fill_lgr_data_structures(
    const arborx::device_point_view&                 search_points,
    const QueryViewType&                             queries,
    const hpc::counting_range<Index1>&               search_indices,
    hpc::device_range_sum<Index1to2Ordinal, Index1>& search_result_ranges,
    hpc::device_vector<Index2, Index1to2Ordinal>&    search_results)
{
  arborx::device_int_view         offsets("offsets", 0);
  arborx::device_int_view         indices("indices", 0);
  hpc::device_vector<int, Index1> index_counts(search_indices.size());
  do_search_and_fill_counts(search_points, queries, search_indices, offsets, indices, index_counts);
  size_and_fill_lgr_data_structures(search_indices, indices, index_counts, search_result_ranges, search_results);
}

}  // namespace

HPC_NOINLINE void
do_otm_point_nearest_node_search(lgr::state& s, int max_support_nodes_per_point)
{
  auto search_nodes  = arborx::create_arborx_nodes(s);
  auto search_points = arborx::create_arborx_points(s);
  auto queries       = arborx::make_nearest_node_queries(search_points, max_support_nodes_per_point);

  do_search_and_fill_lgr_data_structures(
      search_nodes, queries, s.points, s.points_to_point_nodes, s.point_nodes_to_nodes);

  invert_otm_point_node_relations(s);
}

namespace {

template <typename T, typename Range, typename Policy>
HPC_NOINLINE inline T
reduce_min(Policy p, Range& r, T init)
{
  return hpc::transform_reduce(p, r, init, hpc::minimum<T>(), hpc::identity<T>());
}

}  // namespace

void
do_otm_iterative_point_support_search(lgr::state& s, int min_support_nodes_per_point)
{
  auto search_nodes   = arborx::create_arborx_nodes(s);
  auto search_spheres = arborx::create_arborx_point_spheres(s);
  auto queries        = arborx::make_intersect_sphere_queries(search_spheres);

  hpc::device_vector<int, point_index> counts(s.points.size());
  arborx::device_int_view              offsets("offsets", 0);
  arborx::device_int_view              indices("indices", 0);
  int                                  min_nodes_in_support_over_all_points = 0;
  const double                         inflation_factor                     = 1.2;
  while (min_nodes_in_support_over_all_points < min_support_nodes_per_point) {
    arborx::inflate_sphere_query_radii(queries, inflation_factor);

    do_search_and_fill_counts(search_nodes, queries, s.points, offsets, indices, counts);

    min_nodes_in_support_over_all_points = reduce_min(hpc::device_policy(), counts, hpc::numeric_limits<int>::max());
  }

  size_and_fill_lgr_data_structures(s.points, indices, counts, s.points_to_point_nodes, s.point_nodes_to_nodes);

  invert_otm_point_node_relations(s);
}

HPC_NOINLINE void
do_otm_node_nearest_node_search(const lgr::state& s, nearest_neighbors<node_index>& n, int max_nodes_per_node)
{
  auto search_points = arborx::create_arborx_nodes(s);
  auto queries       = arborx::make_nearest_node_queries(search_points, max_nodes_per_node + 1);

  hpc::device_vector<int, point_index> counts(s.nodes.size());
  arborx::device_int_view              offsets("offsets", 0);
  arborx::device_int_view              indices("indices", 0);
  do_search_and_fill_counts(search_points, queries, s.nodes, offsets, indices, counts);
  size_and_fill_lgr_data_structures_from_symmetric_search(
      s.nodes, indices, offsets, counts, n.entities_to_neighbor_ordinals, n.entities_to_neighbors);
}

HPC_NOINLINE void
do_otm_point_nearest_point_search(const lgr::state& s, nearest_neighbors<point_index>& n, int max_points_per_point)
{
  auto search_points = arborx::create_arborx_points(s);
  auto queries       = arborx::make_nearest_node_queries(search_points, max_points_per_point + 1);

  hpc::device_vector<int, point_index> counts(s.points.size());
  arborx::device_int_view              offsets("offsets", 0);
  arborx::device_int_view              indices("indices", 0);
  do_search_and_fill_counts(search_points, queries, s.points, offsets, indices, counts);
  size_and_fill_lgr_data_structures_from_symmetric_search(
      s.points, indices, offsets, counts, n.entities_to_neighbor_ordinals, n.entities_to_neighbors);
}

#else  // ! LGR_ENABLE_SEARCH

namespace {

HPC_NOINLINE void
search_not_enabled_error()
{
  throw std::runtime_error("ArborX search not enabled! Rebuild with LGR_ENABLE_SEARCH=ON.");
}

}  // namespace

void
initialize_otm_search()
{
  search_not_enabled_error();
}

void
finalize_otm_search()
{
  search_not_enabled_error();
}

HPC_NOINLINE void
do_otm_point_nearest_node_search(lgr::state&, int)
{
  search_not_enabled_error();
}

void
do_otm_iterative_point_support_search(lgr::state&, int)
{
  search_not_enabled_error();
}

HPC_NOINLINE void
do_otm_node_nearest_node_search(const lgr::state&, nearest_neighbors<node_index>&, int)
{
  search_not_enabled_error();
}

HPC_NOINLINE void
do_otm_point_nearest_point_search(const lgr::state&, nearest_neighbors<point_index>&, int)
{
  search_not_enabled_error();
}

#endif

}  // namespace search
}  // namespace lgr
