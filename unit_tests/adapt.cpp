#include <gtest/gtest.h>
#include <gtest/internal/gtest-internal.h>
#include <hpc_macros.hpp>
#include <hpc_range_sum.hpp>
#include <hpc_vector.hpp>
#include <lgr_mesh_indices.hpp>
#include <lgr_state.hpp>
#include <otm_adapt.hpp>
#include <otm_distance.hpp>
#include <otm_search.hpp>
#include <otm_search_util.hpp>
#include <unit_tests/otm_unit_mesh.hpp>
#include <unit_tests/unit_arborx_testing_util.hpp>
#include <unit_tests/unit_device_util.hpp>
#include <cassert>

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

template<typename Index>
HPC_NOINLINE inline void fill_adapt_op_and_other_entities(const nearest_neighbors<Index> &n,
    const hpc::counting_range<Index> &range,
    const hpc::length<double> nearest_criterion,
    const hpc::device_vector<hpc::length<double>, Index>& criteria,
    hpc::device_vector<Index, Index> &other_entities,
    hpc::device_vector<adapt_op, Index>& adapt_ops)
{
  auto others = other_entities.begin();
  auto ops = adapt_ops.begin();
  auto neighbor_ords = n.entities_to_neighbor_ordinals.cbegin();
  auto neighbors = n.entities_to_neighbors.cbegin();
  auto crit = criteria.cbegin();
  auto func = [=] HPC_DEVICE (const Index i)
  {
    if (crit[i] > nearest_criterion)
    {
      auto neighbor_range = neighbor_ords[i];
      assert(neighbor_range.size() == 1);
      others[i] = neighbors[neighbor_range[0]];
      ops[i] = adapt_op::SPLIT;
    } else
    {
      others[i] = Index(-1);
      ops[i] = adapt_op::NONE;
    }
  };
  hpc::for_each(hpc::device_policy(), range, func);
}

HPC_NOINLINE inline void evaluate_node_adapt(const state& s, otm_adapt_state& a,
    const hpc::length<double> min_dist)
{
  node_neighbors n;
  search::do_otm_node_nearest_node_search(s, n, 1);
  compute_node_neighbor_squared_distances(s, n, a.node_criteria);
  fill_adapt_op_and_other_entities(n, s.nodes, min_dist, a.node_criteria, a.other_node, a.node_op);
}

HPC_NOINLINE inline void evaluate_point_adapt(const state& s, otm_adapt_state& a,
    const hpc::length<double> min_dist)
{
  point_neighbors n;
  search::do_otm_point_nearest_point_search(s, n, 1);
  compute_point_neighbor_squared_distances(s, n, a.point_criteria);
  fill_adapt_op_and_other_entities(n, s.points, min_dist, a.point_criteria, a.other_point, a.point_op);
}

template <typename Index>
HPC_NOINLINE inline void choose_adapt(const hpc::counting_range<Index>& range,
    const hpc::device_vector<Index, Index>& other_entity,
    hpc::device_vector<adapt_op, Index>& entity_ops,
    hpc::device_vector<Index, Index>& counts) {
  hpc::fill(hpc::device_policy(), counts, Index(1));
  auto others = other_entity.cbegin();
  auto ops = entity_ops.begin();
  auto new_counts = counts.begin();
  auto func = [=] HPC_DEVICE (const Index i)
  {
    auto op = ops[i];
    if (op == adapt_op::NONE) return;
    auto target = others[i];
    auto target_of_target = others[target];
    if (target_of_target == i && target < i)
    {
      // symmetric nearest neighbor relation
      ops[i] = adapt_op::NONE;
      return;
    }
    Index entity_count(-100);
    if (op == adapt_op::SPLIT)
    {
      entity_count = Index(2);
    } else if (op == adapt_op::COLLAPSE)
    {
      entity_count = Index(0);
    }
    new_counts[i] = entity_count;
  };
  hpc::for_each(hpc::device_policy(), range, func);
}

HPC_NOINLINE inline void choose_node_adapt(const state& s, otm_adapt_state &a)
{
  choose_adapt(s.nodes, a.other_node, a.node_op, a.node_counts);
}

HPC_NOINLINE inline void choose_point_adapt(const state& s, otm_adapt_state &a)
{
  choose_adapt(s.points, a.other_point, a.point_op, a.point_counts);
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

  otm_adapt_state a(s);

  constexpr hpc::length<double> min_dist(0.1);
  evaluate_point_adapt(s, a, min_dist);
  choose_point_adapt(s, a);

  check_choose_point_adapt_for_all_split(s, a);
}
