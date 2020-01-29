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

  auto search_nodes = lgr::search::impl::create_arborx_nodes(s);

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

  auto search_points = lgr::search::impl::create_arborx_points(s);

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
