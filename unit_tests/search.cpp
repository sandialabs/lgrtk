#include <ArborX_LinearBVH.hpp>
#include <gtest/gtest.h>
#include <hpc_algorithm.hpp>
#include <hpc_execution.hpp>
#include <hpc_index.hpp>
#include <hpc_macros.hpp>
#include <Kokkos_Core.hpp>
#include <lgr_mesh_indices.hpp>
#include <lgr_state.hpp>
#include <otm_arborx_search_impl.hpp>
#include <otm_search.hpp>
#include <unit_tests/otm_unit_mesh.hpp>
#include <ArborX_Point.hpp>
#include <ArborX_Predicates.hpp>
#include <hpc_range_sum.hpp>

class search : public ::testing::Test
{
  void SetUp() override
  {
    lgr::search::initialize_otm_search();
  }

  void TearDown() override
  {
    lgr::search::finalize_otm_search();
  }
};

TEST_F(search, canInitializeArborXNodesFromOTMNodes)
{
  lgr::state s;

  tetrahedron_single_point(s);

  auto search_nodes = lgr::search::arborx::create_arborx_nodes(s);

  EXPECT_EQ(search_nodes.extent(0), 4);

  auto nodes_to_x = s.x.cbegin();
  auto node_check_func = HPC_DEVICE [=](lgr::node_index node) {
    auto&& lgr_node_coord = nodes_to_x[node].load();
    auto&& search_node_coord = search_nodes(hpc::weaken(node));
    EXPECT_DOUBLE_EQ(lgr_node_coord(0), search_node_coord[0]);
    EXPECT_DOUBLE_EQ(lgr_node_coord(1), search_node_coord[1]);
    EXPECT_DOUBLE_EQ(lgr_node_coord(2), search_node_coord[2]);
  };

  hpc::for_each(hpc::device_policy(), s.nodes, node_check_func);
}

TEST_F(search, canInitializeArborXPointsFromOTMPoints)
{
  lgr::state s;

  tetrahedron_single_point(s);

  auto search_points = lgr::search::arborx::create_arborx_points(s);

  EXPECT_EQ(search_points.extent(0), 1);

  auto points_to_x = s.xm.cbegin();
  auto pt_check_func = HPC_DEVICE [=](lgr::point_index point) {
    auto&& lgr_pt_coord = points_to_x[point].load();
    auto&& search_pt_coord = search_points(hpc::weaken(point));
    EXPECT_DOUBLE_EQ(lgr_pt_coord(0), search_pt_coord[0]);
    EXPECT_DOUBLE_EQ(lgr_pt_coord(1), search_pt_coord[1]);
    EXPECT_DOUBLE_EQ(lgr_pt_coord(2), search_pt_coord[2]);
  };

  hpc::for_each(hpc::device_policy(), s.points, pt_check_func);
}

TEST_F(search, canDoNearestNodePointSearch) {
  lgr::state s;
  tetrahedron_single_point(s);

  auto search_nodes = lgr::search::arborx::create_arborx_nodes(s);
  auto search_points = lgr::search::arborx::create_arborx_points(s);
  const int num_nodes_per_point_to_find = 4;
  auto queries = lgr::search::arborx::make_nearest_node_queries(search_points, num_nodes_per_point_to_find);

  lgr::search::arborx::device_int_view offsets;
  lgr::search::arborx::device_int_view indices;
  Kokkos::tie(offsets, indices) = lgr::search::arborx::do_search(search_nodes, queries);

  int num_points = search_points.extent(0);
  auto nodes_to_x = s.x.cbegin();

  EXPECT_EQ(num_points, 1);

  auto pt_check_func = HPC_DEVICE [=](lgr::point_index point) {
    auto point_begin = offsets(hpc::weaken(point));
    auto point_end = offsets(hpc::weaken(point)+1);
    EXPECT_EQ(point_end - point_begin, 4);
    for (auto j=point_begin; j<point_end; ++j)
    {
      auto search_node_coord = search_nodes(indices(j));
      auto&& lgr_node_coord = nodes_to_x[indices(j)].load();
      EXPECT_DOUBLE_EQ(lgr_node_coord(0), search_node_coord[0]);
      EXPECT_DOUBLE_EQ(lgr_node_coord(1), search_node_coord[1]);
      EXPECT_DOUBLE_EQ(lgr_node_coord(2), search_node_coord[2]);

    }
  };

  hpc::for_each(hpc::device_policy(), s.points, pt_check_func);
}

TEST_F(search, canDoNearestNodePointSearchThroughLGRInterface) {
  lgr::state s;
  tetrahedron_single_point(s);

  hpc::device_vector<lgr::node_index, lgr::point_node_index> supports_to_nodes_before_search(s.supports_to_nodes.size());
  hpc::copy(s.supports_to_nodes, supports_to_nodes_before_search);

  hpc::device_range_sum<lgr::point_node_index, lgr::point_index> points_to_point_nodes;

  lgr::search::do_otm_point_node_search(s, points_to_point_nodes);

  auto points_to_nodes_of_point = points_to_point_nodes.cbegin();
  auto old_supports_to_nodes = supports_to_nodes_before_search.cbegin();
  auto new_supports_to_nodes = s.supports_to_nodes.cbegin();
  auto node_indices = s.points * s.nodes_in_support;
  auto pt_node_check_func = HPC_DEVICE [=](lgr::point_index point) {
    auto point_node_range = points_to_nodes_of_point[point];
    auto point_node_indices = node_indices[point];
    EXPECT_EQ(point_node_range.size(), 4);
    for (auto point_node : point_node_range)
    {
      auto point_node_index = point_node_indices[point_node];
      EXPECT_EQ(old_supports_to_nodes[point_node_index], new_supports_to_nodes[point_node_index]);
    }
  };

  hpc::for_each(hpc::device_policy(), s.points, pt_node_check_func);
}
