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

struct node_neighbors
{
  hpc::device_range_sum<node_index> nodes_to_node_neighbors;
  hpc::device_vector<node_index, node_index> nodes_to_neighbors;

  void resize(const hpc::device_vector<int, node_index>& counts)
  {
    auto const total_node_neighbors = hpc::reduce(hpc::device_policy(), counts, 0);
    nodes_to_node_neighbors.assign_sizes(counts);
    nodes_to_neighbors.resize(total_node_neighbors);
  }
};

void compute_node_node_neighbors(const lgr::state& s,
    node_neighbors& n)
{
  auto nodes_to_node_points = s.nodes_to_node_points.cbegin();
  auto node_points_to_points = s.node_points_to_points.cbegin();
  auto points_to_point_nodes = s.points_to_point_nodes.cbegin();
  auto point_nodes_to_nodes = s.point_nodes_to_nodes.cbegin();

  hpc::device_vector<int, node_index> neighbor_counts(s.nodes.size(), 0);

  auto counts = neighbor_counts.begin();
  auto count_func = [=] HPC_DEVICE (const node_index node)
  {
    for (auto node_point : nodes_to_node_points[node])
    {
      auto point = node_points_to_points[node_point];
      for (auto point_node : points_to_point_nodes[point])
      {
        auto neighbor_node = point_nodes_to_nodes[point_node];
        if (neighbor_node == node) continue;
        auto count = hpc::atomic_ref<int>(counts[neighbor_node]);
        count++;
      }
    }
  };
  hpc::for_each(hpc::device_policy(), s.nodes, count_func);

  n.resize(neighbor_counts);
  hpc::fill(hpc::device_policy(), neighbor_counts, 0);

  auto neighbors = n.nodes_to_neighbors.begin();
  auto neighbor_ranges = n.nodes_to_node_neighbors.cbegin();
  auto fill_func = [=] HPC_DEVICE (const node_index node)
  {
    for (auto node_point : nodes_to_node_points[node])
    {
      auto point = node_points_to_points[node_point];
      for (auto point_node : points_to_point_nodes[point])
      {
        auto neighbor_node = point_nodes_to_nodes[point_node];
        if (neighbor_node == node) continue;
        auto count = hpc::atomic_ref<int>(counts[neighbor_node]);
        int const offset = count++;
        auto node_neighbor_range = neighbor_ranges[neighbor_node];
        neighbors[node_neighbor_range[offset]] = node;
      }
    }
  };
  hpc::for_each(hpc::device_policy(), s.nodes, fill_func);
}

void compute_node_neighbor_squared_distances(const state &s, const node_neighbors &n,
    hpc::device_vector<hpc::length<double>, node_index> &nodes_to_neighbor_squared_distances)
{
  auto nodes_to_x = s.x.cbegin();
  auto total_node_neighbors = n.nodes_to_neighbors.size();
  nodes_to_neighbor_squared_distances.resize(total_node_neighbors);
  auto neighbor_ranges = n.nodes_to_node_neighbors.cbegin();
  auto neighbors = n.nodes_to_neighbors.cbegin();
  auto distances = nodes_to_neighbor_squared_distances.begin();
  auto distance_func = [=] HPC_DEVICE (const node_index node)
  {
    auto x_node = nodes_to_x[node].load();
    for (auto node_neighbor : neighbor_ranges[node])
    {
      auto neighbor = neighbors[node_neighbor];
      auto x_neighbor = nodes_to_x[neighbor].load();
      distances[node_neighbor] = hpc::norm_squared(x_node - x_neighbor);
    }
  };
  hpc::for_each(hpc::device_policy(), s.nodes, distance_func);
}

void check_single_tetrahedron_node_neighbor_squared_distances(const state &s,
    const node_neighbors &n,
    hpc::device_vector<hpc::length<double>, node_index> &nodes_to_neighbor_squared_distances)
{
  ASSERT_EQ(nodes_to_neighbor_squared_distances.size(), 12);

  auto n2n_distances = nodes_to_neighbor_squared_distances.cbegin();
  auto n2n_neighbors = n.nodes_to_node_neighbors.cbegin();
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


