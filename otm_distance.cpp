#include <hpc_execution.hpp>
#include <hpc_macros.hpp>
#include <otm_distance.hpp>

namespace lgr {
namespace search_util {

template<typename Index>
void compute_connected_neighbor_squared_distances(const hpc::counting_range<Index> &indices,
    const hpc::device_array_vector<hpc::position<double>, Index> &positions,
    const nearest_neighbors<Index> &n,
    hpc::device_vector<hpc::length<double>, Index> &squared_distances)
{
  auto x = positions.cbegin();
  auto total_neighbors = n.entities_to_neighbors.size();
  squared_distances.resize(total_neighbors);
  auto neighbor_ranges = n.entities_to_neighbor_ordinals.cbegin();
  auto neighbors = n.entities_to_neighbors.cbegin();
  auto distances = squared_distances.begin();
  auto distance_func = [=] HPC_DEVICE (const Index i)
  {
    auto x_i = x[i].load();
    for (auto neighbor_ord : neighbor_ranges[i])
    {
      auto neighbor = neighbors[neighbor_ord];
      auto x_neighbor = x[neighbor].load();
      distances[neighbor_ord] = hpc::norm_squared(x_i - x_neighbor);
    }
  };
  hpc::for_each(hpc::device_policy(), indices, distance_func);
}

template
void compute_connected_neighbor_squared_distances(const hpc::counting_range<node_index> &indices,
    const hpc::device_array_vector<hpc::position<double>, node_index> &positions,
    const search_util::nearest_neighbors<node_index> &n,
    hpc::device_vector<hpc::length<double>, node_index> &squared_distances);

#ifdef HPC_ENABLE_STRONG_INDICES
template
void compute_connected_neighbor_squared_distances(const hpc::counting_range<point_index> &indices,
    const hpc::device_array_vector<hpc::position<double>, point_index> &positions,
    const search_util::nearest_neighbors<point_index> &n,
    hpc::device_vector<hpc::length<double>, point_index> &squared_distances);
#endif

}
}
