#include <gtest/gtest.h>
#include <gtest/internal/gtest-internal.h>
#include <hpc_array.hpp>
#include <hpc_array_vector.hpp>
#include <hpc_dimensional.hpp>
#include <hpc_execution.hpp>
#include <hpc_functional.hpp>
#include <hpc_macros.hpp>
#include <hpc_numeric.hpp>
#include <hpc_range.hpp>
#include <hpc_range_sum.hpp>
#include <hpc_vector.hpp>
#include <lgr_adapt_util.hpp>
#include <lgr_mesh_indices.hpp>
#include <lgr_state.hpp>
#include <otm_adapt.hpp>
#include <otm_adapt_util.hpp>
#include <otm_distance_util.hpp>
#include <otm_meshless.hpp>
#include <otm_search.hpp>
#include <otm_vtk.hpp>
#include <unit_tests/otm_unit_mesh.hpp>
#include <unit_tests/unit_arborx_testing_util.hpp>
#include <unit_tests/unit_device_util.hpp>

using namespace lgr;
using namespace lgr::search_util;

class adapt: public ::testing::Test
{
  void SetUp() override
  {
    lgr_unit::arborx_testing_singleton::instance();
  }
};

namespace {

inline void set_up_node_and_point_data_for_adapt(state& s) {
  s.nearest_node_neighbor.resize(s.nodes.size());
  s.nearest_node_neighbor_dist.resize(s.nodes.size());
  s.nearest_point_neighbor.resize(s.points.size());
  s.nearest_point_neighbor_dist.resize(s.points.size());

  otm_update_nearest_node_neighbor_distances(s);
  otm_update_nearest_point_neighbor_distances(s);
}

}

TEST_F(adapt, can_create_otm_adapt_state_from_lgr_state)
{
  state s;
  tetrahedron_single_point(s);
  ASSERT_NO_THROW(otm_adapt_state a(s));
}

HPC_NOINLINE inline void expect_all_nodes_are_adaptivity_candidates(const otm_adapt_state& a,
    const state& s) {
  auto ops = a.node_op.cbegin();
  auto other = a.other_node.cbegin();
  auto check_func = DEVICE_TEST (const node_index i) {
    DEVICE_EXPECT_EQ(ops[i], adapt_op::SPLIT);
    DEVICE_EXPECT_NE(other[i], -1);
  };
  unit::test_for_each(hpc::device_policy(), s.nodes, check_func);
}

HPC_NOINLINE inline void expect_all_points_are_adaptivity_candidates(const otm_adapt_state& a,
    const state& s) {
  auto ops = a.point_op.cbegin();
  auto other = a.other_point.cbegin();
  auto check_func = DEVICE_TEST (const point_index i) {
    DEVICE_EXPECT_EQ(ops[i], adapt_op::SPLIT);
    DEVICE_EXPECT_NE(other[i], -1);
  };
  unit::test_for_each(hpc::device_policy(), s.points, check_func);
}

TEST_F(adapt, can_compute_node_nearest_node_neighbor_distances_as_criteria)
{
  state s;
  tetrahedron_single_point(s);

  set_up_node_and_point_data_for_adapt(s);

  otm_adapt_state a(s);

  constexpr hpc::length<double> min_dist(0.1);
  evaluate_node_adapt(s, a, min_dist);

  EXPECT_EQ(a.node_criteria.size(), a.node_counts.size());
  EXPECT_EQ(a.other_node.size(), a.node_counts.size());

  expect_all_nodes_are_adaptivity_candidates(a, s);
}

TEST_F(adapt, can_compute_point_nearest_point_neighbor_distances_as_criteria)
{
  state s;
  two_tetrahedra_two_points(s);

  set_up_node_and_point_data_for_adapt(s);

  otm_adapt_state a(s);

  constexpr hpc::length<double> min_dist(0.1);
  evaluate_point_adapt(s, a, min_dist);

  EXPECT_EQ(a.point_criteria.size(), a.point_counts.size());
  EXPECT_EQ(a.other_point.size(), a.point_counts.size());

  expect_all_points_are_adaptivity_candidates(a, s);
}

template <typename Index>
HPC_NOINLINE inline void check_choose_adapt_for_all_split(const hpc::counting_range<Index>& range,
    const hpc::device_vector<Index, Index>& other_entity,
    const hpc::device_vector<adapt_op, Index>& entity_ops,
    const hpc::device_vector<Index, Index>& counts)
{
  auto ops = entity_ops.cbegin();
  auto others = other_entity.cbegin();
  auto new_counts = counts.cbegin();
  auto check_func = DEVICE_TEST (const Index n) {
    auto op = ops[n];
    auto target = others[n];
    auto count = new_counts[n];
    auto target_of_target = others[target];
    if (target_of_target == n && target < n)
    {
      DEVICE_EXPECT_EQ(op, adapt_op::NONE);
      DEVICE_EXPECT_EQ(count, Index(1));
    } else {
      DEVICE_EXPECT_EQ(op, adapt_op::SPLIT);
      DEVICE_EXPECT_EQ(count, Index(2));
    }
  };
  unit::test_for_each(hpc::device_policy(), range, check_func);
}

HPC_NOINLINE inline void check_choose_node_adapt_for_all_split(const state& s, const otm_adapt_state& a)
{
  check_choose_adapt_for_all_split(s.nodes, a.other_node, a.node_op, a.node_counts);
}

HPC_NOINLINE inline void check_choose_point_adapt_for_all_split(const state& s, const otm_adapt_state& a)
{
  check_choose_adapt_for_all_split(s.points, a.other_point, a.point_op, a.point_counts);
}

TEST_F(adapt, can_choose_correct_node_adaptivity_based_on_nearest_neighbor)
{
  state s;
  tetrahedron_single_point(s);

  set_up_node_and_point_data_for_adapt(s);

  otm_adapt_state a(s);

  constexpr hpc::length<double> min_dist(0.1);
  evaluate_node_adapt(s, a, min_dist);
  choose_node_adapt(s, a);

  check_choose_node_adapt_for_all_split(s, a);
}

TEST_F(adapt, can_choose_correct_point_adaptivity_based_on_nearest_neighbor)
{
  state s;
  two_tetrahedra_two_points(s);

  set_up_node_and_point_data_for_adapt(s);

  otm_adapt_state a(s);

  constexpr hpc::length<double> min_dist(0.1);
  evaluate_point_adapt(s, a, min_dist);
  choose_point_adapt(s, a);

  check_choose_point_adapt_for_all_split(s, a);
}

HPC_NOINLINE inline void check_point_node_connectivities_after_single_tet_node_adapt(const state& s)
{
  ASSERT_EQ(s.nodes.size(), 7);
  auto const nodes_to_node_points = s.nodes_to_node_points.cbegin();
  auto const node_points_to_points = s.node_points_to_points.cbegin();
  auto node_check_func = DEVICE_TEST (const node_index n) {
    auto node_points = nodes_to_node_points[n];
    DEVICE_ASSERT_EQ(node_points.size(), 1);
    for (auto node_point : node_points)
    {
      auto const point = node_points_to_points[node_point];
      DEVICE_EXPECT_EQ(point, point_index(0));
    }
  };
  unit::test_for_each(hpc::device_policy(), s.nodes, node_check_func);

  ASSERT_EQ(s.points.size(), 1);
  auto const points_to_point_nodes = s.points_to_point_nodes.cbegin();
  auto const point_nodes_to_nodes = s.point_nodes_to_nodes.cbegin();
  auto point_check_func = DEVICE_TEST (const point_index p) {
    hpc::array<bool, 7, node_index> node_connect_found {};
    auto const point_nodes = points_to_point_nodes[p];
    DEVICE_ASSERT_EQ(point_nodes.size(), 7);
    for (auto const point_node : point_nodes)
    {
      auto const node = point_nodes_to_nodes[point_node];
      DEVICE_ASSERT_LT(node, node_index(7));
      node_connect_found[node] = true;
    }
    for (const bool found : node_connect_found)
    {
      DEVICE_EXPECT_EQ(found, true);
    }
  };
  unit::test_for_each(hpc::device_policy(), s.points, point_check_func);
}

TEST_F(adapt, can_apply_node_adaptivity)
{
  state s;
  tetrahedron_single_point(s);

  set_up_node_and_point_data_for_adapt(s);

  otm_adapt_state a(s);

  constexpr hpc::length<double> min_dist(0.1);
  evaluate_node_adapt(s, a, min_dist);
  choose_node_adapt(s, a);
  auto const num_chosen = get_num_chosen_for_adapt(a.node_op);
  ASSERT_EQ(num_chosen, 3);

  auto const num_new_nodes = hpc::reduce(hpc::device_policy(), a.node_counts, node_index(0));
  ASSERT_EQ(num_new_nodes, 7);

  hpc::offset_scan(hpc::device_policy(), a.node_counts, a.old_nodes_to_new_nodes);
  a.new_nodes.resize(num_new_nodes);
  a.new_nodes_to_old_nodes.resize(num_new_nodes);
  a.new_nodes_are_same.resize(num_new_nodes);
  a.interpolate_from_nodes.resize(num_new_nodes);
  project(s.nodes, a.old_nodes_to_new_nodes, a.new_nodes_to_old_nodes);

  apply_node_adapt(s, a);
  interpolate_nodal_data(a, s.x);
  s.nodes = a.new_nodes;
  search::do_otm_iterative_point_support_search(s, 4);

  check_point_node_connectivities_after_single_tet_node_adapt(s);
}

HPC_NOINLINE inline void check_point_node_connectivities_after_two_tet_point_adapt(const state& s)
{
  ASSERT_EQ(s.nodes.size(), 5);
  auto const nodes_to_node_points = s.nodes_to_node_points.cbegin();
  auto const node_points_to_points = s.node_points_to_points.cbegin();
  auto node_check_func = DEVICE_TEST (const node_index n) {
    hpc::array<bool, 3, point_index> node_point_found {};
    auto node_points = nodes_to_node_points[n];
    DEVICE_ASSERT_EQ(node_points.size(), 3);
    for (auto node_point : node_points)
    {
      auto const point = node_points_to_points[node_point];
      DEVICE_ASSERT_LT(point, point_index(3));
      node_point_found[point] = true;
    }
    for (const bool found : node_point_found)
    {
      DEVICE_EXPECT_TRUE(found);
    }
  };
  unit::test_for_each(hpc::device_policy(), s.nodes, node_check_func);

  ASSERT_EQ(s.points.size(), 3);
  auto const points_to_point_nodes = s.points_to_point_nodes.cbegin();
  auto const point_nodes_to_nodes = s.point_nodes_to_nodes.cbegin();
  auto point_check_func = DEVICE_TEST (const point_index p) {
    hpc::array<bool, 5, node_index> node_connect_found {};
    auto const point_nodes = points_to_point_nodes[p];
    DEVICE_ASSERT_EQ(point_nodes.size(), 5);
    for (auto const point_node : point_nodes)
    {
      auto const node = point_nodes_to_nodes[point_node];
      DEVICE_ASSERT_LT(node, node_index(5));
      node_connect_found[node] = true;
    }
    for (const bool found : node_connect_found)
    {
      DEVICE_EXPECT_TRUE(found);
    }
  };
  unit::test_for_each(hpc::device_policy(), s.points, point_check_func);
}

TEST_F(adapt, can_apply_point_adaptivity)
{
  state s;
  two_tetrahedra_two_points(s);

  set_up_node_and_point_data_for_adapt(s);

  otm_adapt_state a(s);

  constexpr hpc::length<double> max_allowable_neighbor_dist(0.1);
  evaluate_point_adapt(s, a, max_allowable_neighbor_dist);
  choose_point_adapt(s, a);
  auto const num_chosen = get_num_chosen_for_adapt(a.point_op);
  ASSERT_EQ(num_chosen, 1);

  auto const num_new_points = hpc::reduce(hpc::device_policy(), a.point_counts, point_index(0));
  ASSERT_EQ(num_new_points, 3);

  hpc::offset_scan(hpc::device_policy(), a.point_counts, a.old_points_to_new_points);
  a.new_points.resize(num_new_points);
  a.new_points_to_old_points.resize(num_new_points);
  a.new_points_are_same.resize(num_new_points);
  a.interpolate_from_points.resize(num_new_points);
  project(s.points, a.old_points_to_new_points, a.new_points_to_old_points);

  apply_point_adapt(s, a);
  interpolate_point_data(a, s.xp);
  interpolate_point_data(a, s.h_otm); // TODO: what to do? search depends on length...
  s.points = a.new_points;
  search::do_otm_iterative_point_support_search(s, 5);

  check_point_node_connectivities_after_two_tet_point_adapt(s);
}

TEST_F(adapt, performance_test_adapt)
{
  state s;
  elastic_wave_four_points_per_tetrahedron(s);

  set_up_node_and_point_data_for_adapt(s);

  input in(0, 0);

  otm_allocate_state(in, s);
  auto const b0 = hpc::acceleration<double>::zero();
  hpc::fill(hpc::device_policy(), s.b, b0);
  auto const F0 = hpc::deformation_gradient<double>::identity();
  hpc::fill(hpc::device_policy(), s.F_total, F0);
  auto const Fp0 = hpc::deformation_gradient<double>::identity();
  hpc::fill(hpc::device_policy(), s.Fp_total, Fp0);

  in.enable_adapt = true;
  in.max_node_neighbor_distance = 1.0e-6;
  in.max_point_neighbor_distance = 1.0e-8;

  EXPECT_TRUE(otm_adapt(in, s));
}
