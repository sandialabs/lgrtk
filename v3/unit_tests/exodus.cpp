#include <gtest/gtest.h>
#include <hpc_algorithm.hpp>
#include <hpc_array.hpp>
#include <hpc_execution.hpp>
#include <hpc_index.hpp>
#include <hpc_macros.hpp>
#include <hpc_range.hpp>
#include <hpc_vector.hpp>
#include <lgr_exodus.hpp>
#include <otm_tetrahedron_util.hpp>
#include <lgr_input.hpp>
#include <lgr_mesh_indices.hpp>
#include <lgr_state.hpp>
#include <otm_tet2meshless.hpp>
#include <unit_tests/otm_unit_mesh.hpp>
#include <unit_tests/unit_device_util.hpp>

using namespace lgr;

namespace {
template <typename T> using range_size_type = typename hpc::counting_range<T>::size_type;
using nodes_size_type = range_size_type<node_index>;
using elems_size_type = range_size_type<element_index>;
using points_size_type = range_size_type<point_index>;
using nodes_in_elem_size_type = range_size_type<node_in_element_index>;
using nodes_in_support_size_type = range_size_type<point_node_index>;
using point_nodes_size_type = range_size_type<node_point_index>;
}

TEST(exodus, readSimpleFile)
{
  material_index mat(1);
  material_index bnd(1);
  input in(mat, bnd);
  state st;

  int err_code = read_exodus_file("tets.g", in, st);

  ASSERT_EQ(err_code, 0);

  EXPECT_EQ(st.nodes.size(), nodes_size_type(12));
  EXPECT_EQ(st.elements.size(), elems_size_type(12));
}

hpc::device_vector<node_index, point_node_index> collect_points_to_nodes_from_elements(const state &st)
{
  const auto element_nodes_to_nodes = st.elements_to_nodes.cbegin();
  const auto elements_to_element_nodes = st.elements * st.nodes_in_element;
  const auto elements_to_points = st.elements * st.points_in_element;
  const auto nodes_in_element = st.nodes_in_element;
  const auto points_in_element = st.points_in_element;
  hpc::device_vector<node_index, point_node_index> point_node_indices;
  point_node_indices.resize(st.points.size() * st.nodes_in_element.size());
  auto points_to_point_nodes = st.points * st.nodes_in_element;
  auto points_to_nodes = point_node_indices.begin();
  auto elem_func = [=] HPC_DEVICE (const element_index element)
  {
    const auto element_nodes = elements_to_element_nodes[element];
    const auto element_points = elements_to_points[element];
    for (const auto point_ordinal: points_in_element)
    {
      auto point = element_points[point_ordinal];
      auto point_nodes = points_to_point_nodes[point];
      for (const auto node_ordinal: nodes_in_element)
      {
        const auto node = element_nodes_to_nodes[element_nodes[node_ordinal]];
        const auto point_node = point_nodes[node_ordinal];
        points_to_nodes[hpc::weaken(point_node)] = node;
      }
    }
  };
  hpc::for_each(hpc::device_policy(), st.elements, elem_func);
  return point_node_indices;
}

namespace lgr_unit {

void check_connectivity_sizes(const lgr::state& st, const nodes_in_support_size_type expected_num_support_nodes) {
  auto support_nodes = st.points_to_point_nodes.cbegin();
  auto check_points_func = DEVICE_TEST(point_index const point)
  {
    auto point_support_node_range = support_nodes[point];
    DEVICE_EXPECT_EQ(point_support_node_range.size(), expected_num_support_nodes);
  };
  unit::test_for_each(hpc::device_policy(), st.points, check_points_func);
}

void check_meshfree_against_mesh_connectivities(const lgr::state& st) {
  hpc::device_vector<node_index, point_node_index> expected_point_node_indices =
      collect_points_to_nodes_from_elements(st);
  auto expected_points_to_nodes = expected_point_node_indices.cbegin();
  auto supports = st.points_to_point_nodes.cbegin();
  auto support_nodes_to_nodes = st.point_nodes_to_nodes.cbegin();
  auto pt_func = DEVICE_TEST(point_index const point) {
    auto point_support_nodes = supports[point];
    for (auto n : point_support_nodes)
    {
      DEVICE_EXPECT_EQ(expected_points_to_nodes[hpc::weaken(n)], support_nodes_to_nodes[n]);
    }
  };
  unit::test_for_each(hpc::device_policy(), st.points, pt_func);
}

}

TEST(exodus, convertTetMeshToMeshfree)
{
  material_index mat(1);
  material_index bnd(1);
  input in(mat, bnd);
  state st;
  in.otm_material_points_to_add_per_element = 1;

  in.xp_transform = compute_material_points_as_element_centroids;

  int err_code = read_exodus_file("tets.g", in, st);

  ASSERT_EQ(err_code, 0);

  ASSERT_EQ(st.points.size(), points_size_type(12));
  ASSERT_EQ(st.nodes_in_element.size(), nodes_in_elem_size_type(4));

  convert_tet_mesh_to_meshless(in, st);

  ASSERT_EQ(st.points_to_point_nodes.size(), st.points.size());

  lgr_unit::check_connectivity_sizes(st, 4);
  lgr_unit::check_meshfree_against_mesh_connectivities(st);
}

TEST(exodus, convertTetMeshToMeshfreeInterpolateSingleMaterialPoint) {
  material_index mat(1);
  material_index bnd(1);
  input in(mat, bnd);
  state st;

  auto const points_per_element = 1;
  in.otm_material_points_to_add_per_element = points_per_element;
  lgr::tet_nodes_to_points point_interpolator(points_per_element);
  in.xp_transform = std::ref(point_interpolator);

  int err_code = read_exodus_file("tets.g", in, st);

  ASSERT_EQ(err_code, 0);

  EXPECT_EQ(st.points.size(), points_size_type(12));
  EXPECT_EQ(st.nodes_in_element.size(), nodes_in_elem_size_type(4));

  convert_tet_mesh_to_meshless(in, st);

  EXPECT_EQ(st.points_to_point_nodes.size(), st.points.size());

  lgr_unit::check_connectivity_sizes(st, 4);
  lgr_unit::check_meshfree_against_mesh_connectivities(st);
}

TEST(exodus, convertTetMeshToMeshfreeInterpolateMaterialPoints) {
  material_index mat(1);
  material_index bnd(1);
  input in(mat, bnd);
  state st;

  auto const points_per_element = 4;
  in.otm_material_points_to_add_per_element = points_per_element;
  lgr::tet_nodes_to_points point_interpolator(points_per_element);
  in.xp_transform = std::ref(point_interpolator);

  int err_code = read_exodus_file("tets.g", in, st);

  ASSERT_EQ(err_code, 0);

  EXPECT_EQ(st.points.size(), points_size_type(12));
  EXPECT_EQ(st.nodes_in_element.size(), nodes_in_elem_size_type(4));

  convert_tet_mesh_to_meshless(in, st);

  EXPECT_EQ(st.points_to_point_nodes.size(), st.points.size());
  EXPECT_EQ(st.xp.size(), st.points.size());

  lgr_unit::check_connectivity_sizes(st, 4);
}
