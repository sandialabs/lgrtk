#include <cassert>
#include <hpc_algorithm.hpp>
#include <hpc_dimensional.hpp>
#include <hpc_execution.hpp>
#include <hpc_functional.hpp>
#include <hpc_macros.hpp>
#include <hpc_range.hpp>
#include <hpc_transform_reduce.hpp>
#include <hpc_vector.hpp>
#include <lgr_state.hpp>
#include <limits>
#include <otm_distance.hpp>
#include <otm_distance_util.hpp>
#include <otm_search.hpp>
#include <otm_search_util.hpp>

namespace lgr {

template <typename Index>
HPC_NOINLINE inline void
compute_sqrt_nearest_neighbor_distances(
    const hpc::counting_range<Index>&               range,
    hpc::device_vector<hpc::length<double>, Index>& dists)
{
  assert(dists.size() == range.size());
  auto const neighbor_distances = dists.begin();
  auto       dist_func          = [=] HPC_DEVICE(Index const i) {
    auto const dist       = std::sqrt(neighbor_distances[i]);
    neighbor_distances[i] = dist;
  };
  hpc::for_each(hpc::device_policy(), range, dist_func);
}

template <typename Index>
HPC_NOINLINE inline void
fill_nearest_neighbors(
    const hpc::counting_range<Index>&            range,
    const search_util::nearest_neighbors<Index>& neighbors,
    hpc::device_vector<Index, Index>&            nearest_neighbor)
{
  assert(nearest_neighbor.size() == range.size());
  assert(neighbors.entities_to_neighbors.size() == range.size());
  auto const result          = nearest_neighbor.begin();
  auto const nearest_ordinal = neighbors.entities_to_neighbor_ordinals.cbegin();
  auto const nearest         = neighbors.entities_to_neighbors.cbegin();
  auto       dist_func       = [=] HPC_DEVICE(Index const i) {
    auto const nearest_ord = nearest_ordinal[i];
    assert(nearest_ord.size() == 1);
    result[i] = nearest[nearest_ord[0]];
  };
  hpc::for_each(hpc::device_policy(), range, dist_func);
}

void
otm_update_nearest_point_neighbor_distances(state& s)
{
  hpc::fill(hpc::device_policy(), s.nearest_point_neighbor_dist, -1.0);
  hpc::fill(hpc::device_policy(), s.nearest_point_neighbor, point_index(-1));
  search_util::point_neighbors n;
  search::do_otm_point_nearest_point_search(s, n, 1);

  if (n.entities_to_neighbors.empty()) return;

  search_util::compute_point_neighbor_squared_distances(s, n, s.nearest_point_neighbor_dist);
  compute_sqrt_nearest_neighbor_distances(s.points, s.nearest_point_neighbor_dist);
  fill_nearest_neighbors(s.points, n, s.nearest_point_neighbor);
}

void
otm_update_nearest_node_neighbor_distances(state& s)
{
  hpc::fill(hpc::device_policy(), s.nearest_node_neighbor_dist, -1.0);
  hpc::fill(hpc::device_policy(), s.nearest_node_neighbor, node_index(-1));
  search_util::node_neighbors n;
  search::do_otm_node_nearest_node_search(s, n, 1);

  if (n.entities_to_neighbors.empty()) return;

  search_util::compute_node_neighbor_squared_distances(s, n, s.nearest_node_neighbor_dist);
  compute_sqrt_nearest_neighbor_distances(s.nodes, s.nearest_node_neighbor_dist);
  fill_nearest_neighbors(s.nodes, n, s.nearest_node_neighbor);
}

void
otm_update_min_nearest_neighbor_distances(state& s)
{
  // TODO compute a relative change in distance as a metric...average?
  hpc::length<double> const init = std::numeric_limits<double>::max();
  s.min_node_neighbor_dist       = hpc::transform_reduce(
      hpc::device_policy(),
      s.nearest_node_neighbor_dist,
      init,
      hpc::minimum<hpc::length<double>>(),
      hpc::identity<hpc::length<double>>());
  s.min_point_neighbor_dist = hpc::transform_reduce(
      hpc::device_policy(),
      s.nearest_point_neighbor_dist,
      init,
      hpc::minimum<hpc::length<double>>(),
      hpc::identity<hpc::length<double>>());
}

}  // namespace lgr
