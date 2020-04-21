#pragma once

#include <gtest/gtest.h>
#include <hpc_dimensional.hpp>
#include <hpc_execution.hpp>
#include <hpc_index.hpp>
#include <hpc_range.hpp>
#include <hpc_range_sum.hpp>
#include <hpc_vector.hpp>
#include <lgr_mesh_indices.hpp>
#include <lgr_state.hpp>
#include <otm_search_util.hpp>
#include <unit_tests/unit_device_util.hpp>

inline void check_single_tetrahedron_node_neighbor_squared_distances(const lgr::state &s,
    const lgr::search_util::node_neighbors &n,
    hpc::device_vector<hpc::length<double>, lgr::node_index> &nodes_to_neighbor_squared_distances)
{
  ASSERT_EQ(nodes_to_neighbor_squared_distances.size(), 12);

  auto n2n_distances = nodes_to_neighbor_squared_distances.cbegin();
  auto n2n_neighbors = n.entities_to_neighbor_ordinals.cbegin();
  auto check_func = DEVICE_TEST (lgr::node_index node)
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

inline void check_single_tetrahedron_nearest_node_squared_distance(const lgr::state &s,
    const lgr::search_util::node_neighbors &n,
    hpc::device_vector<hpc::length<double>, lgr::node_index> &nodes_to_neighbor_squared_distances)
{
  ASSERT_EQ(nodes_to_neighbor_squared_distances.size(), 4);

  auto n2n_distances = nodes_to_neighbor_squared_distances.cbegin();
  auto n2n_neighbors = n.entities_to_neighbor_ordinals.cbegin();
  auto check_func = DEVICE_TEST (lgr::node_index node)
  {
    constexpr hpc::length<double> expected = 1.0;
    auto closest_neighbor = n2n_neighbors[node];
    DEVICE_ASSERT_EQ(closest_neighbor.size(), 1);
    auto const node_to_node_dist = n2n_distances[closest_neighbor[0]];
    DEVICE_EXPECT_EQ(expected, node_to_node_dist);
  };
  unit::test_for_each(hpc::device_policy(), s.nodes, check_func);
}

inline void check_two_tetrahedron_point_neighbor_squared_distances(const lgr::state &s,
    const lgr::search_util::point_neighbors &n,
    hpc::device_vector<hpc::length<double>, lgr::point_index> &pts_to_neighbor_squared_distances)
{
  ASSERT_EQ(pts_to_neighbor_squared_distances.size(), 2);

  auto p2p_distances = pts_to_neighbor_squared_distances.cbegin();
  auto p2p_neighbors = n.entities_to_neighbor_ordinals.cbegin();
  auto check_func =
  DEVICE_TEST (lgr::point_index pt)
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


