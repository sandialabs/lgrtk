#pragma once

#include <hpc_execution.hpp>
#include <hpc_range_sum.hpp>
#include <hpc_vector.hpp>
#include <lgr_mesh_indices.hpp>

namespace lgr {
namespace search_util {

template<typename Neighbor>
struct nearest_neighbors
{
  hpc::device_range_sum<Neighbor, Neighbor> entities_to_neighbor_ordinals;
  hpc::device_vector<Neighbor, Neighbor> entities_to_neighbors;

  void resize(const hpc::device_vector<int, Neighbor>& counts)
  {
    auto const total_neighbors = hpc::reduce(hpc::device_policy(), counts, 0);
    entities_to_neighbor_ordinals.assign_sizes(counts);
    entities_to_neighbors.resize(total_neighbors);
  }
};

using node_neighbors = nearest_neighbors<node_index>;
using point_neighbors = nearest_neighbors<point_index>;

}
}
