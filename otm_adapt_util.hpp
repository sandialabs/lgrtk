#pragma once

#include <cassert>
#include <hpc_array.hpp>
#include <hpc_array_vector.hpp>
#include <hpc_dimensional.hpp>
#include <hpc_execution.hpp>
#include <hpc_macros.hpp>
#include <hpc_range.hpp>
#include <hpc_vector.hpp>
#include <lgr_state.hpp>
#include <otm_adapt.hpp>
#include <otm_distance.hpp>
#include <otm_search.hpp>
#include <otm_search_util.hpp>

namespace lgr {

template <typename Index>
HPC_NOINLINE inline void
evaluate_adapt(
    const hpc::device_vector<hpc::length<double>, Index>& nearest_neighbor_dists,
    const hpc::device_vector<Index, Index>&               nearest_neighbors,
    const hpc::counting_range<Index>&                     range,
    const hpc::length<double>                             nearest_criterion,
    hpc::device_vector<hpc::length<double>, Index>&       criteria,
    hpc::device_vector<Index, Index>&                     other_entities,
    hpc::device_vector<adapt_op, Index>&                  adapt_ops)
{
  assert(criteria.size() == range.size());
  auto others         = other_entities.begin();
  auto ops            = adapt_ops.begin();
  auto neighbors      = nearest_neighbors.cbegin();
  auto neighbor_dists = nearest_neighbor_dists.cbegin();
  auto crit           = criteria.begin();
  auto func           = [=] HPC_DEVICE(const Index i) {
    crit[i] = neighbor_dists[i];
    if (crit[i] > nearest_criterion) {
      others[i] = neighbors[i];
      ops[i]    = adapt_op::SPLIT;
    } else {
      others[i] = Index(-1);
      ops[i]    = adapt_op::NONE;
    }
  };
  hpc::for_each(hpc::device_policy(), range, func);
}

HPC_NOINLINE inline void
evaluate_node_adapt(const state& s, otm_adapt_state& a, const hpc::length<double> min_dist)
{
  evaluate_adapt(
      s.nearest_node_neighbor_dist,
      s.nearest_node_neighbor,
      s.nodes,
      min_dist,
      a.node_criteria,
      a.other_node,
      a.node_op);
}

HPC_NOINLINE inline void
evaluate_point_adapt(const state& s, otm_adapt_state& a, const hpc::length<double> min_dist)
{
  search_util::point_neighbors n;
  search::do_otm_point_nearest_point_search(s, n, 1);
  compute_point_neighbor_squared_distances(s, n, a.point_criteria);
  evaluate_adapt(
      s.nearest_point_neighbor_dist,
      s.nearest_point_neighbor,
      s.points,
      min_dist,
      a.point_criteria,
      a.other_point,
      a.point_op);
}

template <typename Index>
HPC_NOINLINE inline void
choose_adapt(
    const hpc::counting_range<Index>&       range,
    const hpc::device_vector<Index, Index>& other_entity,
    hpc::device_vector<adapt_op, Index>&    entity_ops,
    hpc::device_vector<Index, Index>&       counts)
{
  hpc::fill(hpc::device_policy(), counts, Index(1));
  auto others     = other_entity.cbegin();
  auto ops        = entity_ops.begin();
  auto new_counts = counts.begin();
  auto func       = [=] HPC_DEVICE(const Index i) {
    auto op = ops[i];
    if (op == adapt_op::NONE) return;
    auto target           = others[i];
    auto target_of_target = others[target];
    if (target_of_target == i && target < i) {
      // symmetric nearest neighbor relation
      ops[i] = adapt_op::NONE;
      return;
    }
    Index entity_count(-100);
    if (op == adapt_op::SPLIT) {
      entity_count = Index(2);
    } else if (op == adapt_op::COLLAPSE) {
      entity_count = Index(0);
    }
    new_counts[i] = entity_count;
  };
  hpc::for_each(hpc::device_policy(), range, func);
}

HPC_NOINLINE inline void
choose_node_adapt(const state& s, otm_adapt_state& a)
{
  choose_adapt(s.nodes, a.other_node, a.node_op, a.node_counts);
}

HPC_NOINLINE inline void
choose_point_adapt(const state& s, otm_adapt_state& a)
{
  choose_adapt(s.points, a.other_point, a.point_op, a.point_counts);
}

template <typename Index>
HPC_NOINLINE inline int
get_num_chosen_for_adapt(const hpc::device_vector<adapt_op, Index>& ops)
{
  auto const num_chosen =
      hpc::transform_reduce(hpc::device_policy(), ops, int(0), hpc::plus<int>(), [] HPC_DEVICE(adapt_op const op) {
        return op == adapt_op::NONE ? 0 : 1;
      });
  return num_chosen;
}

template <typename Index>
HPC_NOINLINE inline void
apply_adapt(
    const hpc::counting_range<Index>&                           range,
    const hpc::device_vector<adapt_op, Index>&                  ops,
    const hpc::device_vector<Index, Index>&                     others,
    hpc::device_vector<Index, Index>&                           old_to_new,
    hpc::device_vector<bool, Index>&                            new_are_same,
    hpc::device_array_vector<hpc::array<Index, 2, int>, Index>& interpolate_from)
{
  hpc::fill(hpc::device_policy(), new_are_same, true);
  auto const entity_to_op              = ops.cbegin();
  auto const entity_to_other           = others.cbegin();
  auto       old_to_new_entities       = old_to_new.begin();
  auto       new_entities_are_same     = new_are_same.begin();
  auto       interpolate_from_entities = interpolate_from.begin();
  auto       func                      = [=] HPC_DEVICE(Index const i) {
    auto op = entity_to_op[i];
    if (op == adapt_op::NONE) return;
    auto const target = entity_to_other[i];
    if (op == adapt_op::SPLIT) {
      auto const new_entity               = old_to_new_entities[i];
      auto const split_entity             = new_entity + Index(1);
      new_entities_are_same[split_entity] = false;
      hpc::array<Index, 2, int> interp_from;
      interp_from[0]                          = i;
      interp_from[1]                          = target;
      interpolate_from_entities[split_entity] = interp_from;
    }
  };
  hpc::for_each(hpc::device_policy(), range, func);
}

HPC_NOINLINE inline void
apply_node_adapt(const state& s, otm_adapt_state& a)
{
  apply_adapt(
      s.nodes, a.node_op, a.other_node, a.old_nodes_to_new_nodes, a.new_nodes_are_same, a.interpolate_from_nodes);
}

HPC_NOINLINE inline void
apply_point_adapt(const state& s, otm_adapt_state& a)
{
  apply_adapt(
      s.points,
      a.point_op,
      a.other_point,
      a.old_points_to_new_points,
      a.new_points_are_same,
      a.interpolate_from_points);
}

template <class Range>
HPC_NOINLINE inline void
interpolate_nodal_data(const otm_adapt_state& a, Range& data)
{
  interpolate_data(a.new_nodes, a.new_nodes_to_old_nodes, a.new_nodes_are_same, a.interpolate_from_nodes, data);
}

template <class Range>
HPC_NOINLINE inline void
interpolate_point_data(const otm_adapt_state& a, Range& data)
{
  interpolate_data(a.new_points, a.new_points_to_old_points, a.new_points_are_same, a.interpolate_from_points, data);
}

template <class Range>
HPC_NOINLINE inline void
lie_interpolate_point_data(const otm_adapt_state& a, Range& data)
{
  lie_interpolate_data(
      a.new_points, a.new_points_to_old_points, a.new_points_are_same, a.interpolate_from_points, data);
}

template <class Range>
HPC_NOINLINE inline void
distribute_point_data(const otm_adapt_state& a, Range& data)
{
  distribute_data(a.new_points, a.new_points_to_old_points, a.new_points_are_same, a.interpolate_from_points, data);
}

}  // namespace lgr
