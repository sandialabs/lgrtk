#pragma once

#include <hpc_array_vector.hpp>
#include <hpc_dimensional.hpp>
#include <hpc_range.hpp>
#include <hpc_vector.hpp>
#include <lgr_mesh_indices.hpp>

namespace lgr {

template<typename state_type>
inline void resize_state_arrays(state_type &state, const node_index num_nodes,
    const point_index num_points)
{
  state.x.resize(num_nodes);
  state.xp.resize(num_points);

  state.point_nodes_to_nodes.resize(num_points * num_nodes);
  state.node_points_to_points.resize(num_nodes * num_points);
  state.node_points_to_point_nodes.resize(num_nodes * num_points);

  state.V.resize(num_points);
  state.rho.resize(num_points);
}

struct otm_host_pinned_state {
  hpc::pinned_array_vector<hpc::position<double>, node_index> x;
  hpc::pinned_array_vector<hpc::position<double>, point_index> xp;

  hpc::pinned_vector<node_index, point_node_index> point_nodes_to_nodes;
  hpc::pinned_vector<point_index, node_point_index> node_points_to_points;
  hpc::pinned_vector<point_node_index, node_point_index> node_points_to_point_nodes;

  hpc::pinned_vector<hpc::volume<double>, point_index> V;
  hpc::pinned_vector<hpc::density<double>, point_index> rho;
};

}
