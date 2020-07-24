#pragma once

#include <hpc_array_vector.hpp>
#include <hpc_dimensional.hpp>
#include <hpc_range.hpp>
#include <hpc_vector.hpp>
#include <lgr_mesh_indices.hpp>
#include <lgr_state.hpp>
#include <otm_search_util.hpp>

namespace lgr {
namespace search_util {

template <typename Index>
void
compute_connected_neighbor_squared_distances(
    const hpc::counting_range<Index>&                             indices,
    const hpc::device_array_vector<hpc::position<double>, Index>& positions,
    const search_util::nearest_neighbors<Index>&                  n,
    hpc::device_vector<hpc::length<double>, Index>&               squared_distances);

inline void
compute_node_neighbor_squared_distances(
    const state&                                         s,
    const search_util::node_neighbors&                   n,
    hpc::device_vector<hpc::length<double>, node_index>& nodes_to_neighbor_squared_distances)
{
  compute_connected_neighbor_squared_distances(s.nodes, s.x, n, nodes_to_neighbor_squared_distances);
}

inline void
compute_point_neighbor_squared_distances(
    const state&                                          s,
    const search_util::point_neighbors&                   n,
    hpc::device_vector<hpc::length<double>, point_index>& points_to_neighbor_squared_distances)
{
  compute_connected_neighbor_squared_distances(s.points, s.xp, n, points_to_neighbor_squared_distances);
}

}  // namespace search_util
}  // namespace lgr
