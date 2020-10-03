#include <gtest/gtest.h>

#include <hpc_atomic.hpp>
#include <hpc_dimensional.hpp>
#include <hpc_execution.hpp>
#include <hpc_index.hpp>
#include <hpc_macros.hpp>
#include <hpc_range.hpp>
#include <hpc_range_sum.hpp>
#include <hpc_vector.hpp>
#include <lgr_mesh_indices.hpp>
#include <lgr_state.hpp>
#include <otm_distance.hpp>
#include <otm_search_util.hpp>
#include <unit_tests/otm_unit_mesh.hpp>
#include <unit_tests/unit_device_util.hpp>
#include <unit_tests/unit_otm_distance_util.hpp>

using namespace lgr;
using namespace lgr::search_util;

namespace {

template <typename Index1, typename Index2, typename Idx1to2, typename Idx2to1>
void
compute_connected_neighbors(
    const hpc::counting_range<Index1>&            indices1,
    const hpc::device_range_sum<Idx1to2, Index1>& index1_to_index2_ordinal,
    const hpc::device_vector<Index2, Idx1to2>&    index2_ordinal_to_index2,
    const hpc::device_range_sum<Idx2to1, Index2>& index2_to_index1_ordinal,
    const hpc::device_vector<Index1, Idx2to1>&    index1_ordinal_to_index1,
    nearest_neighbors<Index1>&                    n)
{
  auto idx1_to_idx2ord = index1_to_index2_ordinal.cbegin();
  auto idx2ord_to_idx2 = index2_ordinal_to_index2.cbegin();
  auto idx2_to_idx1ord = index2_to_index1_ordinal.cbegin();
  auto idx1ord_to_idx1 = index1_ordinal_to_index1.cbegin();

  hpc::device_vector<int, Index1> neighbor_counts(indices1.size(), 0);
  auto                            counts     = neighbor_counts.begin();
  auto                            count_func = [=] HPC_DEVICE(const Index1 i) {
    for (auto ord2 : idx1_to_idx2ord[i]) {
      auto idx2 = idx2ord_to_idx2[ord2];
      for (auto ord1 : idx2_to_idx1ord[idx2]) {
        auto neighbor = idx1ord_to_idx1[ord1];
        if (neighbor == i) continue;
        auto count = hpc::atomic_ref<int>(counts[neighbor]);
        count++;
      }
    }
  };
  hpc::for_each(hpc::device_policy(), indices1, count_func);

  n.resize(neighbor_counts);
  hpc::fill(hpc::device_policy(), neighbor_counts, 0);

  auto neighbors       = n.entities_to_neighbors.begin();
  auto neighbor_ranges = n.entities_to_neighbor_ordinals.cbegin();
  auto fill_func       = [=] HPC_DEVICE(const Index1 i) {
    for (auto ord2 : idx1_to_idx2ord[i]) {
      auto idx2 = idx2ord_to_idx2[ord2];
      for (auto ord1 : idx2_to_idx1ord[idx2]) {
        auto neighbor = idx1ord_to_idx1[ord1];
        if (neighbor == i) continue;
        auto      count                        = hpc::atomic_ref<int>(counts[neighbor]);
        int const offset                       = count++;
        auto      node_neighbor_range          = neighbor_ranges[neighbor];
        neighbors[node_neighbor_range[offset]] = i;
      }
    }
  };
  hpc::for_each(hpc::device_policy(), indices1, fill_func);

  auto sort_func = [=] HPC_DEVICE(const Index1 i) {
    auto                        neighbor_range = neighbor_ranges[i];
    hpc::counting_range<Index1> except_last(neighbor_range.begin(), neighbor_range.end() - 1);
    for (auto neighbor_ordinal1 : except_last) {
      auto                        neighbor1    = neighbors[neighbor_ordinal1];
      Index1                      min_neighbor = neighbor1;
      hpc::counting_range<Index1> remaining(neighbor_ordinal1, neighbor_range.end());
      for (auto neighbor_ordinal2 : remaining) {
        min_neighbor = hpc::min(min_neighbor, neighbors[neighbor_ordinal2]);
      }
      hpc::swap(neighbor1, min_neighbor);
    }
  };
  hpc::for_each(hpc::device_policy(), indices1, sort_func);

  hpc::device_vector<int, Index1> unique_counts(neighbor_counts.size());
  hpc::copy(neighbor_counts, unique_counts);
  auto uq_counts   = unique_counts.begin();
  auto unique_func = [=] HPC_DEVICE(const Index1 i) {
    auto neighbor_range = neighbor_ranges[i];
    auto last_unique    = neighbor_range.begin();
    auto except_first   = hpc::counting_range<Index1>(neighbor_range.begin() + 1, neighbor_range.end());
    for (auto neighbor_ordinal : except_first) {
      if (neighbors[neighbor_ordinal] != neighbors[*last_unique]) {
        ++last_unique;
        if (*last_unique != neighbor_ordinal) {
          neighbors[*last_unique] = std::move(neighbors[neighbor_ordinal]);
        }
      }
    }
    uq_counts[i] = ++last_unique - neighbor_range.begin();
  };
  hpc::for_each(hpc::device_policy(), indices1, unique_func);

  hpc::pinned_vector<int, Index1> pinned_unique_counts(unique_counts.size());
  hpc::pinned_vector<int, Index1> pinned_old_counts(neighbor_counts.size());
  hpc::copy(neighbor_counts, pinned_old_counts);
  hpc::copy(unique_counts, pinned_unique_counts);
  auto old_host_counts    = pinned_old_counts.cbegin();
  auto unique_host_counts = pinned_unique_counts.cbegin();
  auto resize_func        = [=] HPC_HOST(Index1 i) {
    auto old_begin    = neighbors + old_host_counts[i - 1];
    auto new_begin    = neighbors + unique_host_counts[i - 1];
    auto unique_count = unique_host_counts[i];
    auto to_range     = hpc::make_iterator_range(new_begin, new_begin + unique_count);
    auto from_range   = hpc::make_iterator_range(old_begin, old_begin + unique_count);
    hpc::move(hpc::device_policy(), from_range, to_range);
  };
  hpc::counting_range<Index1> all_but_first_index(indices1.begin() + 1, indices1.end());
  hpc::for_each(hpc::serial_policy(), all_but_first_index, resize_func);

  n.resize(unique_counts);
}

}  // namespace

void
compute_node_node_neighbors(const lgr::state& s, node_neighbors& n)
{
  compute_connected_neighbors(
      s.nodes, s.nodes_to_node_points, s.node_points_to_points, s.points_to_point_nodes, s.point_nodes_to_nodes, n);
}

void
compute_point_point_neighbors(const lgr::state& s, point_neighbors& n)
{
  compute_connected_neighbors(
      s.points, s.points_to_point_nodes, s.point_nodes_to_nodes, s.nodes_to_node_points, s.node_points_to_points, n);
}

TEST(distances, can_compute_nearest_and_farthest_node_to_node)
{
  state s;
  tetrahedron_single_point(s);

  node_neighbors n;
  compute_node_node_neighbors(s, n);

  hpc::device_vector<hpc::length<double>, node_index> nodes_to_neighbor_squared_distances;
  compute_node_neighbor_squared_distances(s, n, nodes_to_neighbor_squared_distances);

  check_single_tetrahedron_node_neighbor_squared_distances(s, n, nodes_to_neighbor_squared_distances);
}

TEST(distances, can_compute_nearest_and_farthest_point_to_point)
{
  state s;
  two_tetrahedra_two_points(s);

  point_neighbors n;
  compute_point_point_neighbors(s, n);

  hpc::device_vector<hpc::length<double>, point_index> points_to_neighbor_squared_distances;
  compute_point_neighbor_squared_distances(s, n, points_to_neighbor_squared_distances);

  check_two_tetrahedron_point_neighbor_squared_distances(s, n, points_to_neighbor_squared_distances);
}

TEST(distances, performance_test_node_to_node_distances)
{
  state s;
  elastic_wave_four_points_per_tetrahedron(s);

  node_neighbors n;
  compute_node_node_neighbors(s, n);

  hpc::device_vector<hpc::length<double>, node_index> nodes_to_neighbor_squared_distances;
  compute_node_neighbor_squared_distances(s, n, nodes_to_neighbor_squared_distances);
}

TEST(distances, performance_test_point_to_point_distances)
{
  state s;
  elastic_wave_four_points_per_tetrahedron(s);

  node_neighbors n;
  compute_point_point_neighbors(s, n);

  hpc::device_vector<hpc::length<double>, point_index> nodes_to_neighbor_squared_distances;
  compute_point_neighbor_squared_distances(s, n, nodes_to_neighbor_squared_distances);
}
