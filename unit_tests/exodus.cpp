#include <gtest/gtest.h>
#include <otm_exodus.hpp>
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

void compute_material_points_as_element_centroids(hpc::counting_range<point_index> const points,
    hpc::device_range_sum<point_node_index, point_index> const &points_to_point_nodes,
    hpc::device_vector<node_index, point_node_index> const &point_nodes_to_nodes,
    hpc::device_array_vector<hpc::position<double>, node_index> const &x,
    hpc::device_array_vector<hpc::position<double>, point_index> &xp)
{
  auto pt_to_pt_nodes = points_to_point_nodes.cbegin();
  auto pt_nodes_to_nodes = point_nodes_to_nodes.cbegin();
  auto x_nodes = x.cbegin();
  auto x_points = xp.begin();
  auto point_func = [=] HPC_DEVICE(const point_index point)
  {
    auto const point_nodes = pt_to_pt_nodes[point];
    hpc::position<double> avg_coord(0., 0., 0.);
    for (auto&& point_node : point_nodes)
    {
      auto const node = pt_nodes_to_nodes[point_node];
      avg_coord += x_nodes[node].load();
    }
    avg_coord /= point_nodes.size();

    x_points[point] = avg_coord;
  };
  hpc::for_each(hpc::device_policy(), points, point_func);
}

auto collect_points_to_nodes_from_elements(state &st)
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
  auto elem_func = [=](const element_index element) 
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
  }
  ;
  hpc::for_each(hpc::device_policy(), st.elements, elem_func);
  return point_node_indices;
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

  EXPECT_EQ(st.points.size(), points_size_type(12));
  EXPECT_EQ(st.nodes_in_element.size(), nodes_in_elem_size_type(4));

  convert_tet_mesh_to_meshless(st, in);

  EXPECT_EQ(st.points_to_point_nodes.size(), st.points.size());

  auto support_nodes = st.points_to_point_nodes.cbegin();
  auto check_points_func = DEVICE_TEST(point_index const point)
  {
    auto point_support_node_range = support_nodes[point];
    DEVICE_EXPECT_EQ(point_support_node_range.size(), nodes_in_support_size_type(4));
  };
  unit::test_for_each(hpc::device_policy(), st.points, check_points_func);

  hpc::device_vector<node_index, point_node_index> point_node_indices =
      collect_points_to_nodes_from_elements(st);

  auto points_to_nodes = point_node_indices.cbegin();
  auto supports = st.points_to_point_nodes.cbegin();
  auto support_nodes_to_nodes = st.point_nodes_to_nodes.cbegin();
  auto pt_func = DEVICE_TEST(point_index const point) {
    auto point_support_nodes = supports[point];
    for (auto&& n : point_support_nodes)
    {
      DEVICE_EXPECT_EQ(points_to_nodes[hpc::weaken(n)], support_nodes_to_nodes[n]);
    }
  };
  unit::test_for_each(hpc::device_policy(), st.points, pt_func);
}

TEST(exodus, convertTetMeshToMeshfreeInterpolateSingleMaterialPoint) {
  material_index mat(1);
  material_index bnd(1);
  input in(mat, bnd);
  state st;
  in.otm_material_points_to_add_per_element = 1;

  hpc::host_vector<hpc::position<double>, point_node_index> tet_gauss_pts(1, { 0.25, 0.25, 0.25 });
  tet_gauss_points_to_material_points point_interpolator(tet_gauss_pts);
  in.xp_transform = std::ref(point_interpolator);

  int err_code = read_exodus_file("tets.g", in, st);

  ASSERT_EQ(err_code, 0);

  EXPECT_EQ(st.points.size(), points_size_type(12));
  EXPECT_EQ(st.nodes_in_element.size(), nodes_in_elem_size_type(4));

  convert_tet_mesh_to_meshless(st, in);

  EXPECT_EQ(st.points_to_point_nodes.size(), st.points.size());

  auto support_nodes = st.points_to_point_nodes.cbegin();
  auto check_points_func = DEVICE_TEST(point_index const point)
  {
    auto point_support_node_range = support_nodes[point];
    DEVICE_EXPECT_EQ(point_support_node_range.size(), nodes_in_support_size_type(4));
  };
  unit::test_for_each(hpc::device_policy(), st.points, check_points_func);

  hpc::device_vector<node_index, point_node_index> point_node_indices =
      collect_points_to_nodes_from_elements(st);

  auto points_to_nodes = point_node_indices.cbegin();
  auto supports = st.points_to_point_nodes.cbegin();
  auto support_nodes_to_nodes = st.point_nodes_to_nodes.cbegin();
  auto pt_func = DEVICE_TEST(point_index const point) {
    auto point_support_nodes = supports[point];
    for (auto&& n : point_support_nodes)
    {
      DEVICE_EXPECT_EQ(points_to_nodes[hpc::weaken(n)], support_nodes_to_nodes[n]);
    }
  };
  unit::test_for_each(hpc::device_policy(), st.points, pt_func);
}

TEST(exodus, convertTetMeshToMeshfreeInterpolateMaterialPoints) {
  material_index mat(1);
  material_index bnd(1);
  input in(mat, bnd);
  state st;
  in.otm_material_points_to_add_per_element = 4;

  hpc::host_vector<hpc::position<double>, point_node_index> tet_gauss_pts(4);
  tet_gauss_pts[0] = { 0.1381966011250105, 0.1381966011250105, 0.1381966011250105 };
  tet_gauss_pts[1] = { 0.5854101966249685, 0.1381966011250105, 0.1381966011250105 };
  tet_gauss_pts[2] = { 0.1381966011250105, 0.5854101966249685, 0.1381966011250105 };
  tet_gauss_pts[3] = { 0.1381966011250105, 0.1381966011250105, 0.5854101966249685 };
  tet_gauss_points_to_material_points point_interpolator(tet_gauss_pts);
  in.xp_transform = std::ref(point_interpolator);

  int err_code = read_exodus_file("tets.g", in, st);

  ASSERT_EQ(err_code, 0);

  EXPECT_EQ(st.points.size(), points_size_type(12));
  EXPECT_EQ(st.nodes_in_element.size(), nodes_in_elem_size_type(4));

  convert_tet_mesh_to_meshless(st, in);

  EXPECT_EQ(st.points_to_point_nodes.size(), st.points.size());
  EXPECT_EQ(st.xp.size(), st.points.size());

  auto support_nodes = st.points_to_point_nodes.cbegin();
  auto check_points_func = DEVICE_TEST(point_index const point)
  {
    auto point_support_node_range = support_nodes[point];
    DEVICE_EXPECT_EQ(point_support_node_range.size(), nodes_in_support_size_type(4));
  };
  unit::test_for_each(hpc::device_policy(), st.points, check_points_func);
}
