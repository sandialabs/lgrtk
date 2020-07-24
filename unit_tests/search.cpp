#include <gtest/gtest.h>

#include <functional>
#include <hpc_array_vector.hpp>
#include <hpc_dimensional.hpp>
#include <hpc_execution.hpp>
#include <hpc_index.hpp>
#include <hpc_range.hpp>
#include <hpc_range_sum.hpp>
#include <hpc_vector.hpp>
#include <hpc_vector3.hpp>
#include <lgr_exodus.hpp>
#include <lgr_input.hpp>
#include <lgr_mesh_indices.hpp>
#include <lgr_state.hpp>
#include <otm_arborx_search_impl.hpp>
#include <otm_distance.hpp>
#include <otm_meshing.hpp>
#include <otm_search.hpp>
#include <otm_search_util.hpp>
#include <otm_tet2meshless.hpp>
#include <unit_tests/otm_unit_mesh.hpp>
#include <unit_tests/unit_arborx_testing_util.hpp>
#include <unit_tests/unit_device_util.hpp>
#include <unit_tests/unit_otm_distance_util.hpp>

using namespace lgr;
using namespace lgr::search_util;

class arborx_search : public ::testing::Test
{
  void
  SetUp() override
  {
    lgr_unit::arborx_testing_singleton::instance();
  }
};

namespace lgr_unit {

template <typename index_type>
void
check_lgr_and_arborx_coords(
    const search::arborx::device_point_view                            arborx_coord,
    const hpc::device_array_vector<hpc::position<double>, index_type>& lgr_coord,
    const hpc::counting_range<index_type>&                             range)
{
  auto i_to_x     = lgr_coord.cbegin();
  auto check_func = DEVICE_TEST(index_type i)
  {
    auto&& lgr_i_coord    = i_to_x[i].load();
    auto&& arborx_i_coord = arborx_coord(hpc::weaken(i));
    DEVICE_EXPECT_EQ(lgr_i_coord(0), arborx_i_coord[0]);
    DEVICE_EXPECT_EQ(lgr_i_coord(1), arborx_i_coord[1]);
    DEVICE_EXPECT_EQ(lgr_i_coord(2), arborx_i_coord[2]);
  };
  unit::test_for_each(hpc::device_policy(), range, check_func);
}

}  // namespace lgr_unit

TEST_F(arborx_search, canInitializeArborXNodesFromOTMNodes)
{
  state s;
  tetrahedron_single_point(s);
  auto search_nodes = search::arborx::create_arborx_nodes(s);
  EXPECT_EQ(search_nodes.extent(0), 4u);

  lgr_unit::check_lgr_and_arborx_coords(search_nodes, s.x, s.nodes);
}

TEST_F(arborx_search, canInitializeArborXPointsFromOTMPoints)
{
  state s;

  tetrahedron_single_point(s);

  auto search_points = search::arborx::create_arborx_points(s);

  EXPECT_EQ(search_points.extent(0), 1u);

  lgr_unit::check_lgr_and_arborx_coords(search_points, s.xp, s.points);
}

namespace lgr_unit {

void
check_arborx_spheres(const lgr::state& s, search::arborx::device_sphere_view search_spheres)
{
  auto points_to_x   = s.xp.cbegin();
  auto points_to_r   = s.h_otm.cbegin();
  auto pt_check_func = DEVICE_TEST(point_index point)
  {
    auto&& lgr_pt_coord  = points_to_x[point].load();
    auto&& lgr_pt_h      = points_to_r[point];
    auto&& search_sphere = search_spheres(hpc::weaken(point));
    auto&& sphere_coord  = search_sphere.centroid();
    auto&& sphere_radius = search_sphere.radius();
    DEVICE_EXPECT_EQ(lgr_pt_coord(0), sphere_coord[0]);
    DEVICE_EXPECT_EQ(lgr_pt_coord(1), sphere_coord[1]);
    DEVICE_EXPECT_EQ(lgr_pt_coord(2), sphere_coord[2]);
    DEVICE_EXPECT_EQ(lgr_pt_h, sphere_radius);
  };
  unit::test_for_each(hpc::device_policy(), s.points, pt_check_func);
}

}  // namespace lgr_unit

TEST_F(arborx_search, canInitializeArborXPointSpheresFromOTMPoints)
{
  state s;

  tetrahedron_single_point(s);

  auto search_points = search::arborx::create_arborx_point_spheres(s);

  EXPECT_EQ(search_points.extent(0), 1u);

  lgr_unit::check_arborx_spheres(s, search_points);
}

namespace lgr_unit {

void
check_point_node_search_results(
    const lgr::state&                 s,
    search::arborx::device_int_view   indices,
    search::arborx::device_int_view   offsets,
    search::arborx::device_point_view search_nodes)
{
  auto nodes_to_x    = s.x.cbegin();
  auto pt_check_func = DEVICE_TEST(point_index point)
  {
    auto point_begin = offsets(hpc::weaken(point));
    auto point_end   = offsets(hpc::weaken(point) + 1);
    DEVICE_EXPECT_EQ(point_end - point_begin, 4);
    for (auto j = point_begin; j < point_end; ++j) {
      auto const search_node_coord = search_nodes(indices(j));
      auto const lgr_node_coord    = nodes_to_x[indices(j)].load();
      DEVICE_EXPECT_EQ(lgr_node_coord(0), search_node_coord[0]);
      DEVICE_EXPECT_EQ(lgr_node_coord(1), search_node_coord[1]);
      DEVICE_EXPECT_EQ(lgr_node_coord(2), search_node_coord[2]);
    }
  };
  unit::test_for_each(hpc::device_policy(), s.points, pt_check_func);
}

}  // namespace lgr_unit

TEST_F(arborx_search, canDoNearestNodePointSearch)
{
  state s;
  tetrahedron_single_point(s);

  auto      search_nodes                = search::arborx::create_arborx_nodes(s);
  auto      search_points               = search::arborx::create_arborx_points(s);
  const int num_nodes_per_point_to_find = 4;
  auto      queries = search::arborx::make_nearest_node_queries(search_points, num_nodes_per_point_to_find);

  search::arborx::device_int_view offsets("offsets", 0);
  search::arborx::device_int_view indices("indices", 0);
  search::arborx::do_search(search_nodes, queries, indices, offsets);

  int num_points = search_points.extent(0);

  EXPECT_EQ(num_points, 1);

  lgr_unit::check_point_node_search_results(s, indices, offsets, search_nodes);
}

TEST_F(arborx_search, canDoNodeIntersectingSphereSearch)
{
  state s;
  tetrahedron_single_point(s);

  auto search_nodes   = search::arborx::create_arborx_nodes(s);
  auto search_spheres = search::arborx::create_arborx_point_spheres(s);
  auto queries        = search::arborx::make_intersect_sphere_queries(search_spheres);

  search::arborx::device_int_view offsets("offsets", 0);
  search::arborx::device_int_view indices("indices", 0);
  search::arborx::do_search(search_nodes, queries, indices, offsets);

  int num_points = search_spheres.extent(0);

  EXPECT_EQ(num_points, 1);

  lgr_unit::check_point_node_search_results(s, indices, offsets, search_nodes);
}

namespace {

void
check_search_results(
    state&                                                  s,
    const hpc::device_vector<node_index, point_node_index>& points_to_supported_nodes_before_search,
    const int                                               num_nodes_in_support_before_search)
{
  auto points_to_nodes_of_point      = s.points_to_point_nodes.cbegin();
  auto old_points_to_supported_nodes = points_to_supported_nodes_before_search.cbegin();
  auto new_points_to_supported_nodes = s.point_nodes_to_nodes.cbegin();
  auto pt_node_check_func            = DEVICE_TEST(lgr::point_index point)
  {
    auto point_node_range = points_to_nodes_of_point[point];
    DEVICE_EXPECT_EQ(point_node_range.size(), num_nodes_in_support_before_search);

    // check all nodes found
    for (auto point_node : point_node_range) {
      auto old_node         = old_points_to_supported_nodes[point_node];
      bool found_point_node = false;
      for (auto new_point_node : point_node_range) {
        auto new_node = new_points_to_supported_nodes[new_point_node];
        if (new_node == old_node) {
          found_point_node = true;
          break;
        }
      }
      DEVICE_EXPECT_TRUE(found_point_node);
    }

    // check no extra nodes found
    for (auto new_point_node : point_node_range) {
      auto new_node              = new_points_to_supported_nodes[new_point_node];
      bool found_extraneous_node = true;
      for (auto point_node : point_node_range) {
        auto old_node = old_points_to_supported_nodes[point_node];
        if (new_node == old_node) {
          found_extraneous_node = false;
          break;
        }
      }
      DEVICE_EXPECT_FALSE(found_extraneous_node);
    }
  };
  unit::test_for_each(hpc::device_policy(), s.points, pt_node_check_func);
}

}  // namespace

TEST_F(arborx_search, canDoNearestNodePointSearchThroughLGRInterface)
{
  state s;
  tetrahedron_single_point(s);

  hpc::device_vector<node_index, point_node_index> points_to_supported_nodes_before_search(
      s.point_nodes_to_nodes.size());
  hpc::copy(s.point_nodes_to_nodes, points_to_supported_nodes_before_search);

  search::do_otm_point_nearest_node_search(s, 4);

  check_search_results(s, points_to_supported_nodes_before_search, 4);
}

TEST_F(arborx_search, canDoNearestNodePointSearchTwoTets)
{
  state s;
  two_tetrahedra_two_points(s);

  hpc::device_vector<node_index, point_node_index> points_to_supported_nodes_before_search(
      s.point_nodes_to_nodes.size());
  hpc::copy(s.point_nodes_to_nodes, points_to_supported_nodes_before_search);

  search::do_otm_point_nearest_node_search(s, 5);

  check_search_results(s, points_to_supported_nodes_before_search, 5);
}

TEST_F(arborx_search, canDoIterativeSphereIntersectSearchTwoTets)
{
  state s;
  two_tetrahedra_two_points(s);

  hpc::device_vector<node_index, point_node_index> points_to_supported_nodes_before_search(
      s.point_nodes_to_nodes.size());
  hpc::copy(s.point_nodes_to_nodes, points_to_supported_nodes_before_search);

  search::do_otm_iterative_point_support_search(s, 5);

  check_search_results(s, points_to_supported_nodes_before_search, 5);
}

TEST_F(arborx_search, canDoIterativeSphereIntersectSearchOnExodusMesh)
{
  material_index mat(1);
  material_index bnd(1);
  input          in(mat, bnd);
  state          st;
  in.otm_material_points_to_add_per_element = 1;

  in.xp_transform = compute_material_points_as_element_centroids;

  int err_code = read_exodus_file("tets.g", in, st);

  ASSERT_EQ(err_code, 0);

  convert_tet_mesh_to_meshless(in, st);

  hpc::device_vector<node_index, point_node_index> points_to_supported_nodes_before_search(
      st.point_nodes_to_nodes.size());
  hpc::copy(st.point_nodes_to_nodes, points_to_supported_nodes_before_search);

  EXPECT_NO_THROW(search::do_otm_iterative_point_support_search(st, 4));

  // TODO: reenable results check for improved search...
  //
  // check_search_results(st, points_to_supported_nodes_before_search, 4);
}

namespace lgr_unit {

void
check_inverse_connectivity(const lgr::state& s, const int expected_num_pts_per_node)
{
  auto points_in_influence        = s.nodes_to_node_points.cbegin();
  auto nodes_to_influenced_points = s.node_points_to_points.cbegin();
  auto node_point_check_func      = DEVICE_TEST(node_index const node)
  {
    auto node_points_range = points_in_influence[node];
    DEVICE_EXPECT_EQ(expected_num_pts_per_node, node_points_range.size());
    int ipt = 0;
    for (auto node_point : node_points_range) {
      DEVICE_EXPECT_EQ(ipt, nodes_to_influenced_points[node_point]);
      ++ipt;
    }
  };
  unit::test_for_each(hpc::device_policy(), s.nodes, node_point_check_func);
}

}  // namespace lgr_unit

TEST_F(arborx_search, invertConnectivityForSingleTet)
{
  state s;
  tetrahedron_single_point(s);

  invert_otm_point_node_relations(s);

  lgr_unit::check_inverse_connectivity(s, 1);
}

TEST_F(arborx_search, invertConnectivityForTwoTets)
{
  state s;
  two_tetrahedra_two_points(s);

  invert_otm_point_node_relations(s);

  lgr_unit::check_inverse_connectivity(s, 2);
}

using distances_search = arborx_search;

TEST_F(distances_search, can_compute_nearest_and_farthest_node_to_node_from_search)
{
  state s;
  tetrahedron_single_point(s);

  node_neighbors n;
  search::do_otm_node_nearest_node_search(s, n, 10);

  hpc::device_vector<hpc::length<double>, node_index> nodes_to_neighbor_squared_distances;
  compute_node_neighbor_squared_distances(s, n, nodes_to_neighbor_squared_distances);

  check_single_tetrahedron_node_neighbor_squared_distances(s, n, nodes_to_neighbor_squared_distances);
}

TEST_F(distances_search, search_with_single_node_always_returns_nearest_node)
{
  state s;
  tetrahedron_single_point(s);

  node_neighbors n;
  search::do_otm_node_nearest_node_search(s, n, 1);

  hpc::device_vector<hpc::length<double>, node_index> nodes_to_neighbor_squared_distances;
  compute_node_neighbor_squared_distances(s, n, nodes_to_neighbor_squared_distances);

  check_single_tetrahedron_nearest_node_squared_distance(s, n, nodes_to_neighbor_squared_distances);
}

TEST_F(distances_search, can_compute_nearest_and_farthest_point_to_point_from_search)
{
  state s;
  two_tetrahedra_two_points(s);

  point_neighbors n;
  search::do_otm_point_nearest_point_search(s, n, 10);

  hpc::device_vector<hpc::length<double>, point_index> points_to_neighbor_squared_distances;
  compute_point_neighbor_squared_distances(s, n, points_to_neighbor_squared_distances);

  check_two_tetrahedron_point_neighbor_squared_distances(s, n, points_to_neighbor_squared_distances);
}

TEST_F(distances_search, performance_test_node_to_node_distances_from_search)
{
  state s;
  elastic_wave_four_points_per_tetrahedron(s);

  node_neighbors n;
  search::do_otm_node_nearest_node_search(s, n, 10);

  hpc::device_vector<hpc::length<double>, node_index> nodes_to_neighbor_squared_distances;
  compute_node_neighbor_squared_distances(s, n, nodes_to_neighbor_squared_distances);
}

TEST_F(distances_search, performance_test_point_to_point_distances_from_search)
{
  state s;
  elastic_wave_four_points_per_tetrahedron(s);

  point_neighbors n;
  search::do_otm_point_nearest_point_search(s, n, 10);

  hpc::device_vector<hpc::length<double>, point_index> points_to_neighbor_squared_distances;
  compute_point_neighbor_squared_distances(s, n, points_to_neighbor_squared_distances);
}
