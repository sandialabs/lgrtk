#include <gtest/gtest.h>
#include <hpc_array.hpp>
#include <hpc_dimensional.hpp>
#include <lgr_mesh_indices.hpp>
#include <lgr_state.hpp>
#include <otm_meshless.hpp>
#include <iostream>

int
main(int ac, char* av[])
{
  ::testing::GTEST_FLAG(print_time) = true;
  ::testing::InitGoogleTest(&ac, av);
  auto const retval = RUN_ALL_TESTS();
  return retval;
}

TEST(maxent, values)
{
  lgr::state s;

  using NI = lgr::node_index;
  auto const num_nodes = NI(4);
  s.x.resize(num_nodes);
  auto const nodes_to_x = s.x.begin();
  nodes_to_x[NI(0)] = hpc::position<double>(0, 0, 0);
  nodes_to_x[NI(1)] = hpc::position<double>(1, 0, 0);
  nodes_to_x[NI(2)] = hpc::position<double>(0, 1, 0);
  nodes_to_x[NI(3)] = hpc::position<double>(0, 0, 1);

  using PI = lgr::point_index;
  auto const num_points = PI(1);
  s.xm.resize(num_points);
  auto const points_to_xm = s.xm.begin();
  points_to_xm[PI(0)] = hpc::position<double>(0.25, 0.25, 0.25);

  using NPI = lgr::point_node_index;
  s.N.resize(NPI(num_nodes));

  s.h_otm.resize(num_points);
  auto const points_to_h = s.h_otm.begin();
  points_to_h[PI(0)] = 1.0;

  s.points.resize(num_points);

  using NSI = lgr::node_in_support_index;
  s.nodes_in_support.resize(NSI(4));

  s.supports_to_nodes.resize(num_points * num_nodes);
  auto const support_nodes_to_nodes = s.supports_to_nodes.begin();
  support_nodes_to_nodes[PI(0)] = NI(0);
  support_nodes_to_nodes[PI(1)] = NI(1);
  support_nodes_to_nodes[PI(2)] = NI(2);
  support_nodes_to_nodes[PI(3)] = NI(3);

  ASSERT_LE(0, 1);
}
