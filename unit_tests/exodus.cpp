#include <gtest/gtest.h>
#include <hpc_algorithm.hpp>
#include <hpc_execution.hpp>
#include <hpc_index.hpp>
#include <hpc_macros.hpp>
#include <hpc_range.hpp>
#include <hpc_vector.hpp>
#include <lgr_exodus.hpp>
#include <lgr_input.hpp>
#include <lgr_mesh_indices.hpp>
#include <lgr_state.hpp>
#include <otm_tet2meshless.hpp>

using namespace lgr;

namespace {
template <typename T> using range_size_type = typename hpc::counting_range<T>::size_type;
using nodes_size_type = range_size_type<node_index>;
using elems_size_type = range_size_type<element_index>;
using points_size_type = range_size_type<point_index>;
using nodes_in_elem_size_type = range_size_type<node_in_element_index>;
using nodes_in_support_size_type = range_size_type<node_in_support_index>;
}

TEST(exodus, readSimpleFile) {
    material_index mat(1);
    material_index bnd(1);
    input in(mat, bnd);
    state st;
    in.element = MESHLESS;

    int err_code = read_exodus_file("tets.g", in, st);

    ASSERT_EQ(err_code, 0);

    EXPECT_EQ(st.nodes.size(), nodes_size_type(12));
    EXPECT_EQ(st.elements.size(), elems_size_type(12));
}

TEST(exodus, convertTetMeshToMeshfree) {
  material_index mat(1);
  material_index bnd(1);
  input in(mat, bnd);
  state st;
  in.element = MESHLESS;

  int err_code = read_exodus_file("tets.g", in, st);

  ASSERT_EQ(err_code, 0);

  EXPECT_EQ(st.points.size(), points_size_type(12));
  EXPECT_EQ(st.nodes_in_element.size(), nodes_in_elem_size_type(4));

  convert_tet_mesh_to_meshless(st);

  EXPECT_EQ(st.nodes_in_support.size(), nodes_in_support_size_type(4));

  auto const element_nodes_to_nodes = st.elements_to_nodes.cbegin();
  auto const elements_to_element_nodes = st.elements * st.nodes_in_element;
  auto const elements_to_points = st.elements * st.points_in_element;
  auto const nodes_in_element = st.nodes_in_element;
  auto const points_in_element = st.points_in_element;

  hpc::device_vector<node_index, point_node_index> point_node_indices;
  point_node_indices.resize(st.points.size() * st.nodes_in_support.size());
  auto points_to_point_nodes = st.points * st.nodes_in_element;
  auto points_to_nodes = point_node_indices.begin();

  auto elem_func = [=] HPC_DEVICE (element_index const element) {
    auto const element_nodes = elements_to_element_nodes[element];
    auto const element_points = elements_to_points[element];
    for (auto const point_ordinal : points_in_element) {
      auto point = element_points[point_ordinal];
      auto point_nodes = points_to_point_nodes[point];
      for (auto const node_ordinal : nodes_in_element) {
        auto const node = element_nodes_to_nodes[element_nodes[node_ordinal]];
        auto const point_node = point_nodes[node_ordinal];
        points_to_nodes[hpc::weaken(point_node)] = node;
      }
    }
  };
  hpc::for_each(hpc::device_policy(), st.elements, elem_func);

  auto nodes_in_support = st.nodes_in_support;
  auto supports = st.points * st.nodes_in_support;
  auto support_nodes_to_nodes = st.points_to_supported_nodes.cbegin();
  auto pt_func = [=] HPC_DEVICE (point_index const point) {
    auto point_support_nodes = supports[point];
    for (auto&& n : nodes_in_support)
    {
      EXPECT_EQ(points_to_nodes[hpc::weaken(point_support_nodes[n])], support_nodes_to_nodes[point_support_nodes[n]]);
    }
  };
  hpc::for_each(hpc::device_policy(), st.points, pt_func);
}
