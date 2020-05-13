#include <hpc_execution.hpp>
#include <hpc_numeric.hpp>
#include <lgr_adapt_util.hpp>
#include <lgr_input.hpp>
#include <lgr_state.hpp>
#include <otm_adapt.hpp>
#include <otm_adapt_util.hpp>
#include <otm_search.hpp>
#include <iostream>

namespace lgr {

otm_adapt_state::otm_adapt_state(state const &s) :
    node_criteria(s.nodes.size()),
    point_criteria(s.points.size()),
    other_node(s.nodes.size()),
    other_point(s.points.size()),
    node_op(s.nodes.size()),
    point_op(s.points.size()),
    point_counts(s.points.size()),
    node_counts(s.nodes.size()),
    old_points_to_new_points(s.points.size() + point_index(1)),
    old_nodes_to_new_nodes(s.nodes.size() + node_index(1)),
    new_points_to_old_points(),
    new_nodes_to_old_nodes(),
    new_point_nodes_to_nodes(),
    new_points_are_same(),
    new_nodes_are_same(),
    interpolate_from_nodes(),
    interpolate_from_points(),
    new_points(point_index(0)),
    new_nodes(node_index(0))
{
}

namespace {

template<typename Index>
void resize_and_project_adapt_data(const hpc::counting_range<Index> &old_range,
    const hpc::device_vector<Index, Index> &new_counts, hpc::counting_range<Index> &new_range,
    hpc::device_vector<Index, Index> &old_to_new, hpc::device_vector<bool, Index> &new_are_same,
    hpc::device_array_vector<hpc::array<Index, 2, int>, Index> &interpolate_from,
    hpc::device_vector<Index, Index> &new_to_old)
{
  auto const num_new = hpc::reduce(hpc::device_policy(), new_counts, Index(0));
  hpc::offset_scan(hpc::device_policy(), new_counts, old_to_new);
  new_range.resize(num_new);
  new_to_old.resize(num_new);
  new_are_same.resize(num_new);
  interpolate_from.resize(num_new);
  project(old_range, old_to_new, new_to_old);
}

}

bool otm_adapt(const input& in, state& s)
{
  otm_adapt_state a(s);

  evaluate_node_adapt(s, a, in.max_node_neighbor_distance);
  evaluate_point_adapt(s, a, in.max_point_neighbor_distance);
  choose_node_adapt(s, a);
  choose_point_adapt(s, a);
  auto const num_chosen_nodes = get_num_chosen_for_adapt(a.node_op);
  auto const num_chosen_points = get_num_chosen_for_adapt(a.point_op);

  if (num_chosen_nodes == 0 && num_chosen_points == 0) return false;

  if (in.output_to_command_line)
  {
    std::cout << "adapting " << num_chosen_nodes << " nodes and " << num_chosen_points << " points" << std::endl;
  }

  resize_and_project_adapt_data(s.nodes, a.node_counts, a.new_nodes, a.old_nodes_to_new_nodes,
      a.new_nodes_are_same, a.interpolate_from_nodes, a.new_nodes_to_old_nodes);
  resize_and_project_adapt_data(s.points, a.point_counts, a.new_points, a.old_points_to_new_points,
      a.new_points_are_same, a.interpolate_from_points, a.new_points_to_old_points);

  apply_node_adapt(s, a);
  apply_point_adapt(s, a);
  interpolate_nodal_data(a, s.x);
  interpolate_point_data(a, s.xp);
  interpolate_point_data(a, s.h_otm);
  s.nodes = a.new_nodes;
  s.points = a.new_points;

  search::do_otm_iterative_point_support_search(s, 4);

  return true;
}

}
