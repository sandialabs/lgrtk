#include <hpc_algorithm.hpp>
#include <hpc_array.hpp>
#include <hpc_execution.hpp>
#include <hpc_macros.hpp>
#include <hpc_range.hpp>
#include <hpc_vector.hpp>
#include <lgr_mesh_indices.hpp>
#include <lgr_state.hpp>

namespace lgr {

void convert_tet_mesh_to_meshless(state& st)
{
  using nodes_in_elem_size_type = hpc::counting_range<node_in_element_index>::size_type;

  st.nodes_in_support.resize(st.nodes_in_element.size());

  auto num_points = st.points.size();
  auto num_nodes_in_support = st.nodes_in_support.size();
  st.supports_to_nodes.resize(num_points * num_nodes_in_support);

  auto supports = st.points * st.nodes_in_support;
  auto nodes_in_support = st.nodes_in_support;

  auto nodes_in_element = st.nodes_in_element;
  auto elements_to_element_nodes = st.elements * st.nodes_in_element;
  auto element_nodes_to_nodes = st.elements_to_nodes.cbegin();
  auto points_in_element = st.points_in_element;
  auto elements_to_points = st.elements * st.points_in_element;

  auto support_nodes_to_nodes = st.supports_to_nodes.begin();

  auto func = [=] HPC_DEVICE (element_index const element)
  {
    auto cur_elem_points = elements_to_points[element];
    auto element_nodes = elements_to_element_nodes[element];

    constexpr nodes_in_elem_size_type max_num_elem_nodes = 10;
    hpc::array<node_index, max_num_elem_nodes> cur_elem_nodes;

    for (auto n : nodes_in_element)
    {
      auto cur_elem_node_offset = element_nodes[n];
      auto node = element_nodes_to_nodes[cur_elem_node_offset];
      cur_elem_nodes[n] = node;
    }

    for (auto&& element_point : points_in_element)
    {
      auto&& point = cur_elem_points[element_point];
      auto&& point_support_nodes = supports[point];
      for (auto&& n : nodes_in_support)
      {
        support_nodes_to_nodes[point_support_nodes[n]] = cur_elem_nodes[n];
      }
    }

  };

  hpc::for_each(hpc::device_policy(), st.elements, func);
}

}
