#include <hpc_functional.hpp>
#include <lgr_composite_inline.hpp>
#include <lgr_composite_tetrahedron.hpp>
#include <lgr_state.hpp>

namespace lgr {

namespace composite_tetrahedron {

HPC_HOST_DEVICE constexpr inline matrix4x4<double>
get_parent_M_inv() noexcept
{
  return matrix4x4<double>(
      96.0, -24.0, -24.0, -24.0, -24.0, 96.0, -24.0, -24.0, -24.0, -24.0, 96.0, -24.0, -24.0, -24.0, -24.0, 96.0);
}

HPC_HOST_DEVICE inline hpc::array<double, 4>
get_DOL(hpc::array<double, 12> O_det, subtet_int_t subtet_int) noexcept
{
  hpc::array<double, 4> DOL;
  for (auto& a : DOL) a = 0.0;
  for (int tet = 0; tet < 12; ++tet) {
    for (int pt = 0; pt < 4; ++pt) {
      DOL[pt] += O_det[tet] * subtet_int[tet][pt];
    }
  }
  return DOL;
}

HPC_HOST_DEVICE inline hpc::array<double, 4>
get_volumes(hpc::array<hpc::vector3<double>, 10> const node_coords) noexcept
{
  // compute the projected |J| times integration weights
  constexpr double      ip_weight = 1.0 / 24.0;
  hpc::array<double, 4> volumes;
  for (auto& a : volumes) a = 0.0;
  hpc::array<hpc::vector3<double>, 4> ref_points;
  get_ref_points(ref_points);
  subtet_int_t sub_tet_int;
  get_subtet_int(sub_tet_int);
  S_t S;
  get_S(S);
  O_t O;
  get_O(node_coords, S, O);
  hpc::array<double, 12> O_det;
  get_O_det(O, O_det);
  auto const DOL          = get_DOL(O_det, sub_tet_int);
  auto const parent_M_inv = get_parent_M_inv();
  for (int pt = 0; pt < 4; ++pt) {
    auto const lambda = get_barycentric(ref_points[pt]);
    for (int l1 = 0; l1 < 4; ++l1) {
      for (int l2 = 0; l2 < 4; ++l2) {
        volumes[pt] += lambda[l1] * parent_M_inv(l1, l2) * DOL[l2];
      }
    }
    volumes[pt] *= ip_weight;
  }
  return volumes;
}

}  // namespace composite_tetrahedron

void
initialize_composite_tetrahedron_V(state& s)
{
  auto const element_nodes_to_nodes    = s.elements_to_nodes.cbegin();
  auto const nodes_to_x                = s.x.cbegin();
  auto const points_to_V               = s.V.begin();
  auto const elements_to_element_nodes = s.elements * s.nodes_in_element;
  auto const elements_to_points        = s.elements * s.points_in_element;
  auto const nodes_in_element          = s.nodes_in_element;
  auto const points_in_element         = s.points_in_element;
  auto       functor                   = [=] HPC_DEVICE(element_index const element) {
    auto const                           element_nodes = elements_to_element_nodes[element];
    hpc::array<hpc::vector3<double>, 10> node_coords;
    for (auto const node_in_element : nodes_in_element) {
      auto const node                           = element_nodes_to_nodes[element_nodes[node_in_element]];
      node_coords[hpc::weaken(node_in_element)] = hpc::vector3<double>(nodes_to_x[node].load());
    }
    auto const volumes = composite_tetrahedron::get_volumes(node_coords);
#ifndef NDEBUG
    for (auto const volume : volumes) {
      assert(volume > 0.0);
    }
#endif
    auto const element_points = elements_to_points[element];
    for (auto const qp : points_in_element) {
      points_to_V[element_points[qp]] = volumes[hpc::weaken(qp)];
    }
  };
  hpc::for_each(hpc::device_policy(), s.elements, functor);
}

}  // namespace lgr
