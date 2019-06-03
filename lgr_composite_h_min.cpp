#include <lgr_composite_tetrahedron.hpp>
#include <lgr_composite_inline.hpp>
#include <hpc_functional.hpp>
#include <lgr_state.hpp>

namespace lgr {

namespace composite_tetrahedron {

HPC_HOST_DEVICE inline hpc::array<hpc::vector3<double>, 4> get_subtet_coords(hpc::array<hpc::vector3<double>, 11> in, int subtet) noexcept {
  hpc::array<hpc::vector3<double>, 4> out;
  switch (subtet) {
    case 0:
      out[0] = in[0];
      out[1] = in[4];
      out[2] = in[6];
      out[3] = in[7];
      break;
    case 1:
      out[0] = in[1];
      out[1] = in[5];
      out[2] = in[4];
      out[3] = in[8];
      break;
    case 2:
      out[0] = in[2];
      out[1] = in[6];
      out[2] = in[5];
      out[3] = in[9];
      break;
    case 3:
      out[0] = in[3];
      out[1] = in[8];
      out[2] = in[7];
      out[3] = in[9];
      break;
    case 4:
      out[0] = in[4];
      out[1] = in[8];
      out[2] = in[5];
      out[3] = in[10];
      break;
    case 5:
      out[0] = in[5];
      out[1] = in[8];
      out[2] = in[9];
      out[3] = in[10];
      break;
    case 6:
      out[0] = in[9];
      out[1] = in[8];
      out[2] = in[7];
      out[3] = in[10];
      break;
    case 7:
      out[0] = in[7];
      out[1] = in[8];
      out[2] = in[4];
      out[3] = in[10];
      break;
    case 8:
      out[0] = in[4];
      out[1] = in[5];
      out[2] = in[6];
      out[3] = in[10];
      break;
    case 9:
      out[0] = in[5];
      out[1] = in[9];
      out[2] = in[6];
      out[3] = in[10];
      break;
    case 10:
      out[0] = in[9];
      out[1] = in[7];
      out[2] = in[6];
      out[3] = in[10];
      break;
    case 11:
      out[0] = in[7];
      out[1] = in[4];
      out[2] = in[6];
      out[3] = in[10];
      break;
  }
  return out;
}

HPC_HOST_DEVICE inline double get_tet_diameter(hpc::array<hpc::vector3<double>, 4> const x) noexcept {
  auto const e10 = x[1] - x[0];
  auto const e20 = x[2] - x[0];
  auto const e30 = x[3] - x[0];
  auto const e21 = x[2] - x[1];
  auto const e31 = x[3] - x[1];
  auto const vol = e30 * cross(e10, e20);
  auto const a0 = norm(cross(e10, e20));
  auto const a1 = norm(cross(e10, e30));
  auto const a2 = norm(cross(e20, e30));
  auto const a3 = norm(cross(e21, e31));
  auto const sa = 0.5 * (a0 + a1 + a2 + a3);
  return (sa > 0.0) ? (vol / sa) : 0.0;
}

HPC_HOST_DEVICE double get_length(hpc::array<hpc::vector3<double>, 10> in) noexcept {
  hpc::array<hpc::vector3<double>, 11> node_coords_with_center;
  for (int i = 0; i < 10; ++i) node_coords_with_center[i] = in[i];
  node_coords_with_center[10] = (in[4] + in[5] + in[6] + in[7] + in[8] + in[9]) / 6.0;
  double min_length = hpc::numeric_limits<double>::max();
  for (int tet = 0; tet < 12; ++tet) {
    auto const x = get_subtet_coords(node_coords_with_center, tet);
    auto const length = get_tet_diameter(x);
    min_length = hpc::min(min_length, length);
  }
  constexpr double magic_number = 2.3;
  return min_length * magic_number;
}

}

void update_composite_tetrahedron_h_min(state& s) {
  auto const element_nodes_to_nodes = s.elements_to_nodes.cbegin();
  auto const nodes_to_x = s.x.cbegin();
  auto const elements_to_h_min = s.h_min.begin();
  auto const elements_to_element_nodes = s.elements * s.nodes_in_element;
  auto const nodes_in_element = s.nodes_in_element;
  auto functor = [=] HPC_DEVICE (element_index const element) {
    auto const element_nodes = elements_to_element_nodes[element];
    hpc::array<hpc::vector3<double>, 10> node_coords;
    for (auto const node_in_element : nodes_in_element) {
      auto const node = element_nodes_to_nodes[element_nodes[node_in_element]];
      node_coords[node_in_element.get()] = nodes_to_x[node].load();
    }
    auto const h_min = composite_tetrahedron::get_length(node_coords);
    elements_to_h_min[element] = h_min;
  };
  hpc::for_each(hpc::device_policy(), s.elements, functor);
}

}
