#include <gtest/gtest.h>
#include <hpc_algorithm.hpp>
#include <hpc_array.hpp>
#include <hpc_array_vector.hpp>
#include <hpc_atomic.hpp>
#include <hpc_dimensional.hpp>
#include <hpc_execution.hpp>
#include <hpc_macros.hpp>
#include <hpc_numeric.hpp>
#include <hpc_range_sum.hpp>
#include <hpc_vector.hpp>
#include <hpc_vector3.hpp>
#include <lgr_mesh_indices.hpp>
#include <lgr_state.hpp>
#include <unit_tests/unit_device_util.hpp>
#include <unit_tests/otm_unit_mesh.hpp>

using namespace lgr;

template<typename Index>
struct connected_neighbors
{
  hpc::device_range_sum<Index> entities_to_neighbor_ordinals;
  hpc::device_vector<Index, Index> entities_to_neighbors;

  void resize(const hpc::device_vector<int, Index>& counts)
  {
    auto const total_neighbors = hpc::reduce(hpc::device_policy(), counts, 0);
    entities_to_neighbor_ordinals.assign_sizes(counts);
    entities_to_neighbors.resize(total_neighbors);
  }
};

namespace
{

template <typename Index1, typename Index2, typename Idx1to2, typename Idx2to1>
void compute_connected_neighbors(const hpc::counting_range<Index1>& indices1,
    const hpc::device_range_sum<Idx1to2, Index1>& index1_to_index2_ordinal,
    const hpc::device_vector<Index2, Idx1to2>& index2_ordinal_to_index2,
    const hpc::device_range_sum<Idx2to1, Index2>& index2_to_index1_ordinal,
    const hpc::device_vector<Index1, Idx2to1>& index1_ordinal_to_index1,
    connected_neighbors<Index1>& n)
{
  auto idx1_to_idx2ord = index1_to_index2_ordinal.cbegin();
  auto idx2ord_to_idx2 = index2_ordinal_to_index2.cbegin();
  auto idx2_to_idx1ord = index2_to_index1_ordinal.cbegin();
  auto idx1ord_to_idx1 = index1_ordinal_to_index1.cbegin();

  hpc::device_vector<int, Index1> neighbor_counts(indices1.size(), 0);
  auto counts = neighbor_counts.begin();
  auto count_func = [=] HPC_DEVICE (const Index1 i)
  {
    for (auto ord2 : idx1_to_idx2ord[i])
    {
      auto idx2 = idx2ord_to_idx2[ord2];
      for (auto ord1 : idx2_to_idx1ord[idx2])
      {
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

  auto neighbors = n.entities_to_neighbors.begin();
  auto neighbor_ranges = n.entities_to_neighbor_ordinals.cbegin();
  auto fill_func = [=] HPC_DEVICE (const Index1 i)
  {
    for (auto ord2 : idx1_to_idx2ord[i])
    {
      auto idx2 = idx2ord_to_idx2[ord2];
      for (auto ord1 : idx2_to_idx1ord[idx2])
      {
        auto neighbor = idx1ord_to_idx1[ord1];
        if (neighbor == i) continue;
        auto count = hpc::atomic_ref<int>(counts[neighbor]);
        int const offset = count++;
        auto node_neighbor_range = neighbor_ranges[neighbor];
        neighbors[node_neighbor_range[offset]] = i;
      }
    }
  };
  hpc::for_each(hpc::device_policy(), indices1, fill_func);

  auto sort_func = [=] HPC_DEVICE(const Index1 i)
  {
    auto neighbor_range = neighbor_ranges[i];
    hpc::counting_range<Index1> except_last(neighbor_range.begin(), neighbor_range.end() - 1);
    for (auto neighbor_ordinal1 : except_last)
    {
      auto neighbor1 = neighbors[neighbor_ordinal1];
      Index1 min_neighbor = neighbor1;
      hpc::counting_range<Index1> remaining(neighbor_ordinal1, neighbor_range.end());
      for (auto neighbor_ordinal2 : remaining)
      {
        min_neighbor = hpc::min(min_neighbor, neighbors[neighbor_ordinal2]);
      }
      hpc::swap(neighbor1, min_neighbor);
    }
  };
  hpc::for_each(hpc::device_policy(), indices1, sort_func);

  hpc::device_vector<int, Index1> unique_counts(neighbor_counts.size());
  hpc::copy(neighbor_counts, unique_counts);
  auto uq_counts = unique_counts.begin();
  auto unique_func = [=] HPC_DEVICE(const Index1 i)
  {
    auto neighbor_range = neighbor_ranges[i];
    auto last_unique = neighbor_range.begin();
    auto except_first = hpc::counting_range<Index1>(neighbor_range.begin() + 1, neighbor_range.end());
    for (auto neighbor_ordinal : except_first)
    {
      if (neighbors[neighbor_ordinal] != neighbors[*last_unique])
      {
        ++last_unique;
        if (*last_unique != neighbor_ordinal)
        {
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
  auto old_host_counts = pinned_old_counts.cbegin();
  auto unique_host_counts = pinned_unique_counts.cbegin();
  auto resize_func = [=] HPC_HOST (Index1 i)
  {
    auto old_begin = neighbors + old_host_counts[i-1];
    auto new_begin = neighbors + unique_host_counts[i-1];
    auto unique_count = unique_host_counts[i];
    auto to_range = hpc::make_iterator_range(new_begin, new_begin + unique_count);
    auto from_range = hpc::make_iterator_range(old_begin, old_begin + unique_count);
    hpc::move(hpc::device_policy(), from_range, to_range);
  };
  hpc::counting_range<Index1> all_but_first_index(indices1.begin()+1, indices1.end());
  hpc::for_each(hpc::serial_policy(), all_but_first_index, resize_func);

  n.resize(unique_counts);
}

template<typename Index>
void compute_connected_neighbor_squared_distances(const hpc::counting_range<Index> &indices,
    const hpc::device_array_vector<hpc::position<double>, Index> &positions,
    const connected_neighbors<Index> &n,
    hpc::device_vector<hpc::length<double>, Index> &squared_distances)
{
  auto x = positions.cbegin();
  auto total_neighbors = n.entities_to_neighbors.size();
  squared_distances.resize(total_neighbors);
  auto neighbor_ranges = n.entities_to_neighbor_ordinals.cbegin();
  auto neighbors = n.entities_to_neighbors.cbegin();
  auto distances = squared_distances.begin();
  auto distance_func = [=] HPC_DEVICE (const node_index i)
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

}

using node_neighbors = connected_neighbors<node_index>;
using point_neighbors = connected_neighbors<point_index>;

void compute_node_node_neighbors(const lgr::state &s, node_neighbors &n)
{
  compute_connected_neighbors(s.nodes, s.nodes_to_node_points, s.node_points_to_points,
      s.points_to_point_nodes, s.point_nodes_to_nodes, n);
}

void compute_node_neighbor_squared_distances(const state &s, const node_neighbors &n,
    hpc::device_vector<hpc::length<double>, node_index> &nodes_to_neighbor_squared_distances)
{
  compute_connected_neighbor_squared_distances(s.nodes, s.x, n,
      nodes_to_neighbor_squared_distances);
}

void compute_point_point_neighbors(const lgr::state &s, point_neighbors &n)
{
  compute_connected_neighbors(s.points, s.points_to_point_nodes, s.point_nodes_to_nodes,
      s.nodes_to_node_points, s.node_points_to_points, n);
}

void compute_point_neighbor_squared_distances(const state &s, const point_neighbors &n,
    hpc::device_vector<hpc::length<double>, point_index> &points_to_neighbor_squared_distances)
{
  compute_connected_neighbor_squared_distances(s.points, s.xp, n,
      points_to_neighbor_squared_distances);
}

void check_single_tetrahedron_node_neighbor_squared_distances(const state &s,
    const node_neighbors &n,
    hpc::device_vector<hpc::length<double>, node_index> &nodes_to_neighbor_squared_distances)
{
  ASSERT_EQ(nodes_to_neighbor_squared_distances.size(), 12);

  auto n2n_distances = nodes_to_neighbor_squared_distances.cbegin();
  auto n2n_neighbors = n.entities_to_neighbor_ordinals.cbegin();
  auto check_func = DEVICE_TEST (node_index node)
  {
    constexpr hpc::length<double> expected[4][3]
    {
      { 1.0, 1.0, 1.0},
      { 1.0, 2.0, 2.0},
      { 1.0, 2.0, 2.0},
      { 1.0, 2.0, 2.0}
    };
    auto node_neighbors = n2n_neighbors[node];
    DEVICE_ASSERT_EQ(node_neighbors.size(), 3);
    for (int i=0; i<3; ++i)
    {
      auto const expected_dist = expected[hpc::weaken(node)][i];
      auto const node_to_node_dist = n2n_distances[node_neighbors[i]];
      DEVICE_EXPECT_EQ(expected_dist, node_to_node_dist);
    }
  };
  unit::test_for_each(hpc::device_policy(), s.nodes, check_func);
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

void check_two_tetrahedron_point_neighbor_squared_distances(const state &s,
    const point_neighbors &n,
    hpc::device_vector<hpc::length<double>, point_index> &pts_to_neighbor_squared_distances)
{
  ASSERT_EQ(pts_to_neighbor_squared_distances.size(), 2);

  auto p2p_distances = pts_to_neighbor_squared_distances.cbegin();
  auto p2p_neighbors = n.entities_to_neighbor_ordinals.cbegin();
  auto check_func =
  DEVICE_TEST (point_index pt)
  {
    constexpr hpc::length<double> expected[2][1]
    {
      { 0.1875},
      { 0.1875}
    };
    auto pt_neighbors = p2p_neighbors[pt];
    DEVICE_ASSERT_EQ(pt_neighbors.size(), 1);
    auto const expected_dist = expected[hpc::weaken(pt)][0];
    auto const node_to_node_dist = p2p_distances[pt_neighbors[0]];
    DEVICE_EXPECT_EQ(expected_dist, node_to_node_dist);
  };
  unit::test_for_each(hpc::device_policy(), s.points, check_func);
}

TEST(distances, can_compute_nearest_and_farthest_point_to_point)
{
  state s;
  two_tetrahedra_two_points(s);

  point_neighbors n;
  compute_point_point_neighbors(s, n);

  hpc::device_vector<hpc::length<double>, point_index> nodes_to_neighbor_squared_distances;
  compute_point_neighbor_squared_distances(s, n, nodes_to_neighbor_squared_distances);

  check_two_tetrahedron_point_neighbor_squared_distances(s, n, nodes_to_neighbor_squared_distances);
}
