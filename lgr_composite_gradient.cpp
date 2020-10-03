#include <lgr_composite_inline.hpp>
#include <lgr_composite_tetrahedron.hpp>
#include <lgr_state.hpp>

namespace lgr {

namespace composite_tetrahedron {

HPC_NOINLINE HPC_HOST_DEVICE inline void
get_O_inv(O_t const& O, O_t& O_inv) noexcept
{
  for (int tet = 0; tet < 12; ++tet) {
    O_inv[tet] = inverse(O[tet]);
  }
}

HPC_NOINLINE HPC_HOST_DEVICE inline void
get_subtet_proj_M(subtet_proj_t& sub_tet_int_proj_M) noexcept
{
  sub_tet_int_proj_M[0](0, 0)  = 0.008333333333333333;
  sub_tet_int_proj_M[0](0, 1)  = 0.0015625;
  sub_tet_int_proj_M[0](0, 2)  = 0.0015625;
  sub_tet_int_proj_M[0](0, 3)  = 0.0015625;
  sub_tet_int_proj_M[0](1, 0)  = 0.0015625;
  sub_tet_int_proj_M[0](1, 1)  = 0.0005208333333333333;
  sub_tet_int_proj_M[0](1, 2)  = 0.00026041666666666666;
  sub_tet_int_proj_M[0](1, 3)  = 0.00026041666666666666;
  sub_tet_int_proj_M[0](2, 0)  = 0.0015625;
  sub_tet_int_proj_M[0](2, 1)  = 0.00026041666666666666;
  sub_tet_int_proj_M[0](2, 2)  = 0.0005208333333333333;
  sub_tet_int_proj_M[0](2, 3)  = 0.00026041666666666666;
  sub_tet_int_proj_M[0](3, 0)  = 0.0015625;
  sub_tet_int_proj_M[0](3, 1)  = 0.00026041666666666666;
  sub_tet_int_proj_M[0](3, 2)  = 0.00026041666666666666;
  sub_tet_int_proj_M[0](3, 3)  = 0.0005208333333333333;
  sub_tet_int_proj_M[1](0, 0)  = 0.0005208333333333333;
  sub_tet_int_proj_M[1](0, 1)  = 0.0015625;
  sub_tet_int_proj_M[1](0, 2)  = 0.00026041666666666666;
  sub_tet_int_proj_M[1](0, 3)  = 0.00026041666666666666;
  sub_tet_int_proj_M[1](1, 0)  = 0.0015625;
  sub_tet_int_proj_M[1](1, 1)  = 0.008333333333333333;
  sub_tet_int_proj_M[1](1, 2)  = 0.0015625;
  sub_tet_int_proj_M[1](1, 3)  = 0.0015625;
  sub_tet_int_proj_M[1](2, 0)  = 0.00026041666666666666;
  sub_tet_int_proj_M[1](2, 1)  = 0.0015625;
  sub_tet_int_proj_M[1](2, 2)  = 0.0005208333333333333;
  sub_tet_int_proj_M[1](2, 3)  = 0.00026041666666666666;
  sub_tet_int_proj_M[1](3, 0)  = 0.00026041666666666666;
  sub_tet_int_proj_M[1](3, 1)  = 0.0015625;
  sub_tet_int_proj_M[1](3, 2)  = 0.00026041666666666666;
  sub_tet_int_proj_M[1](3, 3)  = 0.0005208333333333333;
  sub_tet_int_proj_M[2](0, 0)  = 0.0005208333333333333;
  sub_tet_int_proj_M[2](0, 1)  = 0.00026041666666666666;
  sub_tet_int_proj_M[2](0, 2)  = 0.0015625;
  sub_tet_int_proj_M[2](0, 3)  = 0.00026041666666666666;
  sub_tet_int_proj_M[2](1, 0)  = 0.00026041666666666666;
  sub_tet_int_proj_M[2](1, 1)  = 0.0005208333333333333;
  sub_tet_int_proj_M[2](1, 2)  = 0.0015625;
  sub_tet_int_proj_M[2](1, 3)  = 0.00026041666666666666;
  sub_tet_int_proj_M[2](2, 0)  = 0.0015625;
  sub_tet_int_proj_M[2](2, 1)  = 0.0015625;
  sub_tet_int_proj_M[2](2, 2)  = 0.008333333333333333;
  sub_tet_int_proj_M[2](2, 3)  = 0.0015625;
  sub_tet_int_proj_M[2](3, 0)  = 0.00026041666666666666;
  sub_tet_int_proj_M[2](3, 1)  = 0.00026041666666666666;
  sub_tet_int_proj_M[2](3, 2)  = 0.0015625;
  sub_tet_int_proj_M[2](3, 3)  = 0.0005208333333333333;
  sub_tet_int_proj_M[3](0, 0)  = 0.0005208333333333333;
  sub_tet_int_proj_M[3](0, 1)  = 0.00026041666666666666;
  sub_tet_int_proj_M[3](0, 2)  = 0.00026041666666666666;
  sub_tet_int_proj_M[3](0, 3)  = 0.0015625;
  sub_tet_int_proj_M[3](1, 0)  = 0.00026041666666666666;
  sub_tet_int_proj_M[3](1, 1)  = 0.0005208333333333333;
  sub_tet_int_proj_M[3](1, 2)  = 0.00026041666666666666;
  sub_tet_int_proj_M[3](1, 3)  = 0.0015625;
  sub_tet_int_proj_M[3](2, 0)  = 0.00026041666666666666;
  sub_tet_int_proj_M[3](2, 1)  = 0.00026041666666666666;
  sub_tet_int_proj_M[3](2, 2)  = 0.0005208333333333333;
  sub_tet_int_proj_M[3](2, 3)  = 0.0015625;
  sub_tet_int_proj_M[3](3, 0)  = 0.0015625;
  sub_tet_int_proj_M[3](3, 1)  = 0.0015625;
  sub_tet_int_proj_M[3](3, 2)  = 0.0015625;
  sub_tet_int_proj_M[3](3, 3)  = 0.008333333333333333;
  sub_tet_int_proj_M[4](0, 0)  = 0.0004557291666666667;
  sub_tet_int_proj_M[4](0, 1)  = 0.0008463541666666667;
  sub_tet_int_proj_M[4](0, 2)  = 0.0003255208333333333;
  sub_tet_int_proj_M[4](0, 3)  = 0.0003255208333333333;
  sub_tet_int_proj_M[4](1, 0)  = 0.0008463541666666667;
  sub_tet_int_proj_M[4](1, 1)  = 0.002018229166666667;
  sub_tet_int_proj_M[4](1, 2)  = 0.0008463541666666667;
  sub_tet_int_proj_M[4](1, 3)  = 0.0008463541666666667;
  sub_tet_int_proj_M[4](2, 0)  = 0.0003255208333333333;
  sub_tet_int_proj_M[4](2, 1)  = 0.0008463541666666667;
  sub_tet_int_proj_M[4](2, 2)  = 0.0004557291666666667;
  sub_tet_int_proj_M[4](2, 3)  = 0.0003255208333333333;
  sub_tet_int_proj_M[4](3, 0)  = 0.0003255208333333333;
  sub_tet_int_proj_M[4](3, 1)  = 0.0008463541666666667;
  sub_tet_int_proj_M[4](3, 2)  = 0.0003255208333333333;
  sub_tet_int_proj_M[4](3, 3)  = 0.0004557291666666667;
  sub_tet_int_proj_M[5](0, 0)  = 0.00006510416666666667;
  sub_tet_int_proj_M[5](0, 1)  = 0.0001953125;
  sub_tet_int_proj_M[5](0, 2)  = 0.0001953125;
  sub_tet_int_proj_M[5](0, 3)  = 0.0001953125;
  sub_tet_int_proj_M[5](1, 0)  = 0.0001953125;
  sub_tet_int_proj_M[5](1, 1)  = 0.0011067708333333333;
  sub_tet_int_proj_M[5](1, 2)  = 0.0009765625;
  sub_tet_int_proj_M[5](1, 3)  = 0.0009765625;
  sub_tet_int_proj_M[5](2, 0)  = 0.0001953125;
  sub_tet_int_proj_M[5](2, 1)  = 0.0009765625;
  sub_tet_int_proj_M[5](2, 2)  = 0.0011067708333333333;
  sub_tet_int_proj_M[5](2, 3)  = 0.0009765625;
  sub_tet_int_proj_M[5](3, 0)  = 0.0001953125;
  sub_tet_int_proj_M[5](3, 1)  = 0.0009765625;
  sub_tet_int_proj_M[5](3, 2)  = 0.0009765625;
  sub_tet_int_proj_M[5](3, 3)  = 0.0011067708333333333;
  sub_tet_int_proj_M[6](0, 0)  = 0.0004557291666666667;
  sub_tet_int_proj_M[6](0, 1)  = 0.0003255208333333333;
  sub_tet_int_proj_M[6](0, 2)  = 0.0003255208333333333;
  sub_tet_int_proj_M[6](0, 3)  = 0.0008463541666666667;
  sub_tet_int_proj_M[6](1, 0)  = 0.0003255208333333333;
  sub_tet_int_proj_M[6](1, 1)  = 0.0004557291666666667;
  sub_tet_int_proj_M[6](1, 2)  = 0.0003255208333333333;
  sub_tet_int_proj_M[6](1, 3)  = 0.0008463541666666667;
  sub_tet_int_proj_M[6](2, 0)  = 0.0003255208333333333;
  sub_tet_int_proj_M[6](2, 1)  = 0.0003255208333333333;
  sub_tet_int_proj_M[6](2, 2)  = 0.0004557291666666667;
  sub_tet_int_proj_M[6](2, 3)  = 0.0008463541666666667;
  sub_tet_int_proj_M[6](3, 0)  = 0.0008463541666666667;
  sub_tet_int_proj_M[6](3, 1)  = 0.0008463541666666667;
  sub_tet_int_proj_M[6](3, 2)  = 0.0008463541666666667;
  sub_tet_int_proj_M[6](3, 3)  = 0.002018229166666667;
  sub_tet_int_proj_M[7](0, 0)  = 0.0011067708333333333;
  sub_tet_int_proj_M[7](0, 1)  = 0.0009765625;
  sub_tet_int_proj_M[7](0, 2)  = 0.0001953125;
  sub_tet_int_proj_M[7](0, 3)  = 0.0009765625;
  sub_tet_int_proj_M[7](1, 0)  = 0.0009765625;
  sub_tet_int_proj_M[7](1, 1)  = 0.0011067708333333333;
  sub_tet_int_proj_M[7](1, 2)  = 0.0001953125;
  sub_tet_int_proj_M[7](1, 3)  = 0.0009765625;
  sub_tet_int_proj_M[7](2, 0)  = 0.0001953125;
  sub_tet_int_proj_M[7](2, 1)  = 0.0001953125;
  sub_tet_int_proj_M[7](2, 2)  = 0.00006510416666666667;
  sub_tet_int_proj_M[7](2, 3)  = 0.0001953125;
  sub_tet_int_proj_M[7](3, 0)  = 0.0009765625;
  sub_tet_int_proj_M[7](3, 1)  = 0.0009765625;
  sub_tet_int_proj_M[7](3, 2)  = 0.0001953125;
  sub_tet_int_proj_M[7](3, 3)  = 0.0011067708333333333;
  sub_tet_int_proj_M[8](0, 0)  = 0.0011067708333333333;
  sub_tet_int_proj_M[8](0, 1)  = 0.0009765625;
  sub_tet_int_proj_M[8](0, 2)  = 0.0009765625;
  sub_tet_int_proj_M[8](0, 3)  = 0.0001953125;
  sub_tet_int_proj_M[8](1, 0)  = 0.0009765625;
  sub_tet_int_proj_M[8](1, 1)  = 0.0011067708333333333;
  sub_tet_int_proj_M[8](1, 2)  = 0.0009765625;
  sub_tet_int_proj_M[8](1, 3)  = 0.0001953125;
  sub_tet_int_proj_M[8](2, 0)  = 0.0009765625;
  sub_tet_int_proj_M[8](2, 1)  = 0.0009765625;
  sub_tet_int_proj_M[8](2, 2)  = 0.0011067708333333333;
  sub_tet_int_proj_M[8](2, 3)  = 0.0001953125;
  sub_tet_int_proj_M[8](3, 0)  = 0.0001953125;
  sub_tet_int_proj_M[8](3, 1)  = 0.0001953125;
  sub_tet_int_proj_M[8](3, 2)  = 0.0001953125;
  sub_tet_int_proj_M[8](3, 3)  = 0.00006510416666666667;
  sub_tet_int_proj_M[9](0, 0)  = 0.0004557291666666667;
  sub_tet_int_proj_M[9](0, 1)  = 0.0003255208333333333;
  sub_tet_int_proj_M[9](0, 2)  = 0.0008463541666666667;
  sub_tet_int_proj_M[9](0, 3)  = 0.0003255208333333333;
  sub_tet_int_proj_M[9](1, 0)  = 0.0003255208333333333;
  sub_tet_int_proj_M[9](1, 1)  = 0.0004557291666666667;
  sub_tet_int_proj_M[9](1, 2)  = 0.0008463541666666667;
  sub_tet_int_proj_M[9](1, 3)  = 0.0003255208333333333;
  sub_tet_int_proj_M[9](2, 0)  = 0.0008463541666666667;
  sub_tet_int_proj_M[9](2, 1)  = 0.0008463541666666667;
  sub_tet_int_proj_M[9](2, 2)  = 0.002018229166666667;
  sub_tet_int_proj_M[9](2, 3)  = 0.0008463541666666667;
  sub_tet_int_proj_M[9](3, 0)  = 0.0003255208333333333;
  sub_tet_int_proj_M[9](3, 1)  = 0.0003255208333333333;
  sub_tet_int_proj_M[9](3, 2)  = 0.0008463541666666667;
  sub_tet_int_proj_M[9](3, 3)  = 0.0004557291666666667;
  sub_tet_int_proj_M[10](0, 0) = 0.0011067708333333333;
  sub_tet_int_proj_M[10](0, 1) = 0.0001953125;
  sub_tet_int_proj_M[10](0, 2) = 0.0009765625;
  sub_tet_int_proj_M[10](0, 3) = 0.0009765625;
  sub_tet_int_proj_M[10](1, 0) = 0.0001953125;
  sub_tet_int_proj_M[10](1, 1) = 0.00006510416666666667;
  sub_tet_int_proj_M[10](1, 2) = 0.0001953125;
  sub_tet_int_proj_M[10](1, 3) = 0.0001953125;
  sub_tet_int_proj_M[10](2, 0) = 0.0009765625;
  sub_tet_int_proj_M[10](2, 1) = 0.0001953125;
  sub_tet_int_proj_M[10](2, 2) = 0.0011067708333333333;
  sub_tet_int_proj_M[10](2, 3) = 0.0009765625;
  sub_tet_int_proj_M[10](3, 0) = 0.0009765625;
  sub_tet_int_proj_M[10](3, 1) = 0.0001953125;
  sub_tet_int_proj_M[10](3, 2) = 0.0009765625;
  sub_tet_int_proj_M[10](3, 3) = 0.0011067708333333333;
  sub_tet_int_proj_M[11](0, 0) = 0.002018229166666667;
  sub_tet_int_proj_M[11](0, 1) = 0.0008463541666666667;
  sub_tet_int_proj_M[11](0, 2) = 0.0008463541666666667;
  sub_tet_int_proj_M[11](0, 3) = 0.0008463541666666667;
  sub_tet_int_proj_M[11](1, 0) = 0.0008463541666666667;
  sub_tet_int_proj_M[11](1, 1) = 0.0004557291666666667;
  sub_tet_int_proj_M[11](1, 2) = 0.0003255208333333333;
  sub_tet_int_proj_M[11](1, 3) = 0.0003255208333333333;
  sub_tet_int_proj_M[11](2, 0) = 0.0008463541666666667;
  sub_tet_int_proj_M[11](2, 1) = 0.0003255208333333333;
  sub_tet_int_proj_M[11](2, 2) = 0.0004557291666666667;
  sub_tet_int_proj_M[11](2, 3) = 0.0003255208333333333;
  sub_tet_int_proj_M[11](3, 0) = 0.0008463541666666667;
  sub_tet_int_proj_M[11](3, 1) = 0.0003255208333333333;
  sub_tet_int_proj_M[11](3, 2) = 0.0003255208333333333;
  sub_tet_int_proj_M[11](3, 3) = 0.0004557291666666667;
}

HPC_NOINLINE HPC_HOST_DEVICE inline void
get_M_inv(hpc::array<double, 12> const& O_det, matrix4x4<double>& M_inv) noexcept
{
  auto          M = matrix4x4<double>::zero();
  subtet_proj_t sub_tet_int_proj_M;
  get_subtet_proj_M(sub_tet_int_proj_M);
  for (int tet = 0; tet < 12; ++tet) {
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 4; ++j) {
        M(i, j) += O_det[tet] * sub_tet_int_proj_M[tet](i, j);
      }
    }
  }
  M_inv = inverse(M);
}

HPC_NOINLINE HPC_HOST_DEVICE inline void
get_SOL(
    hpc::array<double, 12> const& O_det,
    O_t const&                    O_inv,
    subtet_int_t const&           subtet_int,
    S_t const&                    S,
    SOL_t&                        SOL) noexcept
{
  for (auto& a : SOL)
    for (auto& b : a) b = hpc::vector3<double>::zero();
  for (int tet = 0; tet < 12; ++tet) {
    for (int node = 0; node < 10; ++node) {
      for (int dim1 = 0; dim1 < 3; ++dim1) {
        for (int dim2 = 0; dim2 < 3; ++dim2) {
          for (int pt = 0; pt < 4; ++pt) {
            SOL[pt][node](dim1) += O_det[tet] * S[tet][node](dim2) * O_inv[tet](dim2, dim1) * subtet_int[tet][pt];
          }
        }
      }
    }
  }
}

HPC_NOINLINE HPC_HOST_DEVICE inline void
get_basis_gradients(
    hpc::array<hpc::vector3<double>, 10> const&          node_coords,
    hpc::array<hpc::array<hpc::vector3<double>, 10>, 4>& grad_N) noexcept
{
  for (auto& a : grad_N) {
    for (auto& b : a) {
      b = hpc::vector3<double>::zero();
    }
  }
  hpc::array<hpc::vector3<double>, 4> ref_points;
  get_ref_points(ref_points);
  subtet_int_t subtet_int;
  get_subtet_int(subtet_int);
  S_t S;
  get_S(S);
  O_t O;
  get_O(node_coords, S, O);
  O_t O_inv;
  get_O_inv(O, O_inv);
  hpc::array<double, 12> O_det;
  get_O_det(O, O_det);
  matrix4x4<double> M_inv;
  get_M_inv(O_det, M_inv);
  SOL_t SOL;
  get_SOL(O_det, O_inv, subtet_int, S, SOL);
  for (int node = 0; node < 10; ++node) {
    for (int pt = 0; pt < 4; ++pt) {
      auto const lambda = get_barycentric(ref_points[pt]);
      for (int d = 0; d < 3; ++d) {
        for (int l1 = 0; l1 < 4; ++l1) {
          for (int l2 = 0; l2 < 4; ++l2) {
            grad_N[pt][node](d) += lambda[l1] * M_inv(l1, l2) * SOL[l2][node](d);
          }
        }
      }
    }
  }
}

}  // namespace composite_tetrahedron

void
initialize_composite_tetrahedron_grad_N(state& s)
{
  auto const element_nodes_to_nodes    = s.elements_to_nodes.cbegin();
  auto const nodes_to_x                = s.x.cbegin();
  auto const point_nodes_to_grad_N     = s.grad_N.begin();
  auto const elements_to_element_nodes = s.elements * s.nodes_in_element;
  auto const elements_to_points        = s.elements * s.points_in_element;
  auto const points_to_point_nodes     = s.points * s.nodes_in_element;
  auto const nodes_in_element          = s.nodes_in_element;
  auto const points_in_element         = s.points_in_element;
  auto       functor                   = [=] HPC_DEVICE(element_index const element) {
    auto const                           element_nodes = elements_to_element_nodes[element];
    hpc::array<hpc::vector3<double>, 10> node_coords;
    for (auto const node_in_element : nodes_in_element) {
      auto const node                           = element_nodes_to_nodes[element_nodes[node_in_element]];
      node_coords[hpc::weaken(node_in_element)] = hpc::vector3<double>(nodes_to_x[node].load());
    }
    hpc::array<hpc::array<hpc::vector3<double>, 10>, 4> grad_N;
    composite_tetrahedron::get_basis_gradients(node_coords, grad_N);
    auto const element_points = elements_to_points[element];
    for (auto const qp : points_in_element) {
      auto const point       = element_points[qp];
      auto const point_nodes = points_to_point_nodes[point];
      for (auto const a : nodes_in_element) {
        auto const point_node             = point_nodes[a];
        point_nodes_to_grad_N[point_node] = hpc::basis_gradient<double>(grad_N[hpc::weaken(qp)][hpc::weaken(a)]);
      }
    }
  };
  hpc::for_each(hpc::device_policy(), s.elements, functor);
}

}  // namespace lgr
