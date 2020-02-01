#include <hpc_algorithm.hpp>
#include <hpc_array.hpp>
#include <hpc_array_vector.hpp>
#include <hpc_dimensional.hpp>
#include <hpc_execution.hpp>
#include <hpc_functional.hpp>
#include <hpc_macros.hpp>
#include <hpc_range.hpp>
#include <hpc_vector.hpp>
#include <hpc_vector3.hpp>
#include <lgr_mesh_indices.hpp>
#include <lgr_state.hpp>

namespace lgr {

void convert_tet_mesh_to_meshless(state& st)
{
  auto num_points = st.points.size();
  auto num_nodes_in_support = st.nodes_in_element.size();
  st.points_to_supported_nodes.resize(num_points * num_nodes_in_support);
  st.xm.resize(num_points);
  st.h_otm.resize(num_points);

  auto nodes_in_element = st.nodes_in_element;
  auto elements_to_element_nodes = st.elements * st.nodes_in_element;
  auto element_nodes_to_nodes = st.elements_to_nodes.cbegin();
  auto points_in_element = st.points_in_element;
  auto elements_to_points = st.elements * st.points_in_element;

  hpc::device_vector<int, point_index> nodes_in_support_counts(st.points.size(),
      st.nodes_in_element.size());
  st.nodes_in_support.assign_sizes(nodes_in_support_counts);

  auto support_nodes_to_nodes = st.points_to_supported_nodes.begin();

  auto nodes_to_x = st.x.cbegin();
  auto mat_pts_to_x = st.xm.begin();
  auto mat_pts_to_h = st.h_otm.begin();
  auto nodes_in_support = st.nodes_in_support.cbegin();
  auto func = [=] HPC_DEVICE (element_index const element)
  {
    auto cur_elem_points = elements_to_points[element];
    auto element_nodes = elements_to_element_nodes[element];

    for (auto&& element_point : points_in_element)
    {
      hpc::position<double> avg_coord(0., 0., 0.);
      auto point = cur_elem_points[element_point];
      auto&& point_support_nodes = nodes_in_support[point];
      for (auto&& n : nodes_in_element)
      {
        auto cur_elem_node_offset = element_nodes[n];
        auto node = element_nodes_to_nodes[cur_elem_node_offset];
        avg_coord += nodes_to_x[node].load();
        support_nodes_to_nodes[point_support_nodes[n]] = node;
      }

      avg_coord /= nodes_in_element.size();

      auto min_node_centroid_dist = 1. / hpc::machine_epsilon<double>();
      for (auto&& n : nodes_in_element)
      {
        auto cur_elem_node_offset = element_nodes[n];
        auto node = element_nodes_to_nodes[cur_elem_node_offset];
        auto node_centroid_dist = hpc::norm(nodes_to_x[node].load() - avg_coord);
        min_node_centroid_dist = hpc::min(min_node_centroid_dist, node_centroid_dist);
      }
      mat_pts_to_x[point] = avg_coord;
      mat_pts_to_h[point] = min_node_centroid_dist;
    }
  };

  hpc::for_each(hpc::device_policy(), st.elements, func);
}

}
