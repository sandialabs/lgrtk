#pragma once

#include <hpc_array.hpp>
#include <hpc_array_traits.hpp>
#include <hpc_array_vector.hpp>
#include <hpc_dimensional.hpp>
#include <hpc_macros.hpp>
#include <hpc_matrix3x3.hpp>
#include <hpc_quaternion.hpp>
#include <hpc_range.hpp>
#include <hpc_symmetric3x3.hpp>
#include <hpc_vector.hpp>
#include <hpc_vector3.hpp>
#include <lgr_mesh_indices.hpp>
#include <lgr_state.hpp>
#include <cmath>

namespace lgr {
class state;
}

namespace lgr {

void otm_populate_new_nodes(state & s,
    node_index begin_src, node_index end_src,
    node_index begin_target, node_index end_target);

void otm_populate_new_points(state & s,
    point_index begin_src, point_index end_src,
    point_index begin_target, point_index end_target);

bool otm_adapt(const input& in, state& s);

enum adapt_op {
  NONE,
  SPLIT,
  COLLAPSE,
};

struct otm_adapt_state {
  hpc::device_vector<hpc::length<double>, node_index> node_criteria;
  hpc::device_vector<hpc::length<double>, node_index> point_criteria;
  hpc::device_vector<node_index, node_index> other_node;
  hpc::device_vector<point_index, point_index> other_point;
  hpc::device_vector<adapt_op, node_index> node_op;
  hpc::device_vector<adapt_op, point_index> point_op;
  hpc::device_vector<point_index, point_index> point_counts;
  hpc::device_vector<node_index, node_index> node_counts;
  hpc::device_vector<point_index, point_index> old_points_to_new_points;
  hpc::device_vector<node_index, node_index> old_nodes_to_new_nodes;
  hpc::device_vector<point_index, point_index> new_points_to_old_points;
  hpc::device_vector<node_index, node_index> new_nodes_to_old_nodes;
  hpc::device_vector<node_index, point_node_index> new_point_nodes_to_nodes;
  hpc::device_vector<bool, point_index> new_points_are_same;
  hpc::device_vector<bool, node_index> new_nodes_are_same;
  hpc::device_array_vector<hpc::array<node_index, 2, int>, node_index> interpolate_from_nodes;
  hpc::device_array_vector<hpc::array<point_index, 2, int>, point_index> interpolate_from_points;
  hpc::counting_range<point_index> new_points;
  hpc::counting_range<node_index> new_nodes;

  otm_adapt_state(state const&);
};

} // namespace lgr
