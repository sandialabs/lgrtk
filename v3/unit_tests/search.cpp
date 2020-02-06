#include <gtest/gtest.h>
#include <hpc_algorithm.hpp>
#include <hpc_execution.hpp>
#include <hpc_index.hpp>
#include <hpc_macros.hpp>
#include <hpc_vector.hpp>
#include <Kokkos_Pair.hpp>
#include <Kokkos_View.hpp>
#include <lgr_exodus.hpp>
#include <lgr_input.hpp>
#include <lgr_mesh_indices.hpp>
#include <lgr_state.hpp>
#include <otm_arborx_search_impl.hpp>
#include <otm_meshing.hpp>
#include <otm_search.hpp>
#include <otm_tet2meshless.hpp>
#include <unit_tests/otm_unit_mesh.hpp>

using namespace lgr;

class arborx_search: public ::testing::Test
{
  void SetUp() override
  {
    search::initialize_otm_search();
  }

  void TearDown() override
  {
    search::finalize_otm_search();
  }
};

TEST_F(arborx_search, canInitializeArborXNodesFromOTMNodes)
{
  state s;

  tetrahedron_single_point(s);

  auto search_nodes = search::arborx::create_arborx_nodes(s);

  EXPECT_EQ(search_nodes.extent(0), 4);

  auto nodes_to_x = s.x.cbegin();
  auto node_check_func = [=](node_index node)
  {
    auto&& lgr_node_coord = nodes_to_x[node].load();
    auto&& search_node_coord = search_nodes(hpc::weaken(node));
    EXPECT_DOUBLE_EQ(lgr_node_coord(0), search_node_coord[0]);
    EXPECT_DOUBLE_EQ(lgr_node_coord(1), search_node_coord[1]);
    EXPECT_DOUBLE_EQ(lgr_node_coord(2), search_node_coord[2]);
  };

  hpc::for_each(hpc::device_policy(), s.nodes, node_check_func);
}

TEST_F(arborx_search, canInitializeArborXPointsFromOTMPoints)
{
  state s;

  tetrahedron_single_point(s);

  auto search_points = search::arborx::create_arborx_points(s);

  EXPECT_EQ(search_points.extent(0), 1);

  auto points_to_x = s.xm.cbegin();
  auto pt_check_func = [=](point_index point)
  {
    auto&& lgr_pt_coord = points_to_x[point].load();
    auto&& search_pt_coord = search_points(hpc::weaken(point));
    EXPECT_DOUBLE_EQ(lgr_pt_coord(0), search_pt_coord[0]);
    EXPECT_DOUBLE_EQ(lgr_pt_coord(1), search_pt_coord[1]);
    EXPECT_DOUBLE_EQ(lgr_pt_coord(2), search_pt_coord[2]);
  };

  hpc::for_each(hpc::device_policy(), s.points, pt_check_func);
}

TEST_F(arborx_search, canInitializeArborXPointSpheresFromOTMPoints)
{
  state s;

  tetrahedron_single_point(s);

  auto search_points = search::arborx::create_arborx_point_spheres(s);

  EXPECT_EQ(search_points.extent(0), 1);

  auto points_to_x = s.xm.cbegin();
  auto points_to_r = s.h_otm.cbegin();
  auto pt_check_func = [=](point_index point)
  {
    auto&& lgr_pt_coord = points_to_x[point].load();
    auto&& lgr_pt_h = points_to_r[point];
    auto&& search_sphere = search_points(hpc::weaken(point));
    auto&& sphere_coord = search_sphere.centroid();
    auto&& sphere_radius = search_sphere.radius();
    EXPECT_DOUBLE_EQ(lgr_pt_coord(0), sphere_coord[0]);
    EXPECT_DOUBLE_EQ(lgr_pt_coord(1), sphere_coord[1]);
    EXPECT_DOUBLE_EQ(lgr_pt_coord(2), sphere_coord[2]);
    EXPECT_DOUBLE_EQ(lgr_pt_h, sphere_radius);
  };
  hpc::for_each(hpc::device_policy(), s.points, pt_check_func);
}

TEST_F(arborx_search, canDoNearestNodePointSearch)
{
  state s;
  tetrahedron_single_point(s);

  auto search_nodes = search::arborx::create_arborx_nodes(s);
  auto search_points = search::arborx::create_arborx_points(s);
  const int num_nodes_per_point_to_find = 4;
  auto queries = search::arborx::make_nearest_node_queries(search_points,
      num_nodes_per_point_to_find);

  search::arborx::device_int_view offsets;
  search::arborx::device_int_view indices;
  Kokkos::tie(offsets, indices) = search::arborx::do_search(search_nodes, queries);

  int num_points = search_points.extent(0);
  auto nodes_to_x = s.x.cbegin();

  EXPECT_EQ(num_points, 1);

  auto pt_check_func = [=](point_index point)
  {
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

TEST_F(arborx_search, canDoNodeIntersectingSphereSearch)
{
  state s;
  tetrahedron_single_point(s);

  auto search_nodes = search::arborx::create_arborx_nodes(s);
  auto search_spheres = search::arborx::create_arborx_point_spheres(s);
  auto queries = search::arborx::make_intersect_sphere_queries(search_spheres);

  search::arborx::device_int_view offsets;
  search::arborx::device_int_view indices;
  Kokkos::tie(offsets, indices) = search::arborx::do_search(search_nodes, queries);

  int num_points = search_spheres.extent(0);
  auto nodes_to_x = s.x.cbegin();

  EXPECT_EQ(num_points, 1);

  auto pt_check_func = [=](point_index point)
  {
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

namespace {

void check_search_results(state &s,
    const hpc::device_vector<node_index, point_node_index> &points_to_supported_nodes_before_search,
    const int num_nodes_in_support_before_search)
{
  auto points_to_nodes_of_point = s.points_to_point_nodes.cbegin();
  auto old_points_to_supported_nodes = points_to_supported_nodes_before_search.cbegin();
  auto new_points_to_supported_nodes = s.point_nodes_to_nodes.cbegin();
  auto pt_node_check_func = [=](lgr::point_index point) 
  {
    auto point_node_range = points_to_nodes_of_point[point];
    EXPECT_EQ(point_node_range.size(), num_nodes_in_support_before_search);

    // check all nodes found
    for (auto point_node: point_node_range)
    {
      auto old_node = old_points_to_supported_nodes[point_node];
      bool found_point_node = false;
      for (auto new_point_node: point_node_range)
      {
        auto new_node = new_points_to_supported_nodes[new_point_node];
        if (new_node == old_node)
        {
          found_point_node = true;
          break;
        }
      }
      EXPECT_TRUE(found_point_node) << "  node = " << old_node;
    }

    // check no extra nodes found
    for (auto new_point_node: point_node_range)
    {
      auto new_node = new_points_to_supported_nodes[new_point_node];
      bool found_extraneous_node = true;
      for (auto point_node: point_node_range)
      {
        auto old_node = old_points_to_supported_nodes[point_node];
        if (new_node == old_node)
        {
          found_extraneous_node = false;
          break;
        }
      }
      EXPECT_FALSE(found_extraneous_node) << "  node = " << new_node << std::endl;
    }
  };
  hpc::for_each(hpc::device_policy(), s.points, pt_node_check_func);
}

}

TEST_F(arborx_search, canDoNearestNodePointSearchThroughLGRInterface)
{
  state s;
  tetrahedron_single_point(s);

  hpc::device_vector<node_index, point_node_index> points_to_supported_nodes_before_search(s.point_nodes_to_nodes.size());
  hpc::copy(s.point_nodes_to_nodes, points_to_supported_nodes_before_search);

  search::do_otm_point_nearest_node_search(s, 4);

  check_search_results(s, points_to_supported_nodes_before_search, 4);
}

TEST_F(arborx_search, canDoNearestNodePointSearchTwoTets)
{
  state s;
  two_tetrahedra_two_points(s);

  hpc::device_vector<node_index, point_node_index> points_to_supported_nodes_before_search(s.point_nodes_to_nodes.size());
  hpc::copy(s.point_nodes_to_nodes, points_to_supported_nodes_before_search);

  search::do_otm_point_nearest_node_search(s, 5);

  check_search_results(s, points_to_supported_nodes_before_search, 5);
}

TEST_F(arborx_search, canDoIterativeSphereIntersectSearchTwoTets)
{
  state s;
  two_tetrahedra_two_points(s);

  hpc::device_vector<node_index, point_node_index> points_to_supported_nodes_before_search(s.point_nodes_to_nodes.size());
  hpc::copy(s.point_nodes_to_nodes, points_to_supported_nodes_before_search);

  search::do_otm_iterative_point_support_search(s, 5);

  check_search_results(s, points_to_supported_nodes_before_search, 5);
}

TEST_F(arborx_search, canDoIterativeSphereIntersectSearchOnExodusMesh)
{
  material_index mat(1);
  material_index bnd(1);
  input in(mat, bnd);
  state st;
  in.element = MESHLESS;

  int err_code = read_exodus_file("tets.g", in, st);

  ASSERT_EQ(err_code, 0);

  convert_tet_mesh_to_meshless(st);

  hpc::device_vector<node_index, point_node_index> points_to_supported_nodes_before_search(
      st.point_nodes_to_nodes.size());
  hpc::copy(st.point_nodes_to_nodes, points_to_supported_nodes_before_search);

  EXPECT_NO_THROW(search::do_otm_iterative_point_support_search(st, 4));

  // TODO: reenable results check for improved search...
  //
  // check_search_results(st, points_to_supported_nodes_before_search, 4);
}

TEST_F(arborx_search, invertConnectivityForSingleTet)
{
  state s;
  tetrahedron_single_point(s);

  invert_otm_point_node_relations(s);

  auto points_in_influence = s.nodes_to_node_points.cbegin();
  auto nodes_to_influenced_points = s.node_points_to_points.cbegin();
  auto node_point_check_func = [=] (node_index const node) {
    auto node_points_range = points_in_influence[node];
    EXPECT_EQ(1, node_points_range.size());
    for (auto node_point : node_points_range)
    {
      EXPECT_EQ(0, nodes_to_influenced_points[node_point]);
    }
  };
  hpc::for_each(hpc::device_policy(), s.nodes, node_point_check_func);
}

TEST_F(arborx_search, invertConnectivityForTwoTets)
{
  state s;
  two_tetrahedra_two_points(s);

  invert_otm_point_node_relations(s);

  auto points_in_influence = s.nodes_to_node_points.cbegin();
  auto nodes_to_influenced_points = s.node_points_to_points.cbegin();
  auto node_point_check_func = [=] (node_index const node) {
    auto node_points_range = points_in_influence[node];
    EXPECT_EQ(2, node_points_range.size());
    for (auto i=0; i < 2; ++i)
    {
      auto node_point = node_points_range[i];
      EXPECT_EQ(i, hpc::weaken(nodes_to_influenced_points[node_point]));
    }
  };
  hpc::for_each(hpc::device_policy(), s.nodes, node_point_check_func);
}
