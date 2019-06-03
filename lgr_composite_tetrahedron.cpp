#include <lgr_composite_tetrahedron.hpp>
#include <lgr_composite_inline.hpp>
#include <hpc_functional.hpp>
#include <lgr_state.hpp>

namespace lgr {

namespace composite_tetrahedron {

HPC_HOST_DEVICE inline subtet_proj_t get_subtet_proj_M() noexcept {
  subtet_proj_t sub_tet_int_proj_M;
  sub_tet_int_proj_M[0](0, 0) = 0.008333333333333333;
  sub_tet_int_proj_M[0](0, 1) = 0.0015625;
  sub_tet_int_proj_M[0](0, 2) = 0.0015625;
  sub_tet_int_proj_M[0](0, 3) = 0.0015625;
  sub_tet_int_proj_M[0](1, 0) = 0.0015625;
  sub_tet_int_proj_M[0](1, 1) = 0.0005208333333333333;
  sub_tet_int_proj_M[0](1, 2) = 0.00026041666666666666;
  sub_tet_int_proj_M[0](1, 3) = 0.00026041666666666666;
  sub_tet_int_proj_M[0](2, 0) = 0.0015625;
  sub_tet_int_proj_M[0](2, 1) = 0.00026041666666666666;
  sub_tet_int_proj_M[0](2, 2) = 0.0005208333333333333;
  sub_tet_int_proj_M[0](2, 3) = 0.00026041666666666666;
  sub_tet_int_proj_M[0](3, 0) = 0.0015625;
  sub_tet_int_proj_M[0](3, 1) = 0.00026041666666666666;
  sub_tet_int_proj_M[0](3, 2) = 0.00026041666666666666;
  sub_tet_int_proj_M[0](3, 3) = 0.0005208333333333333;
  sub_tet_int_proj_M[1](0, 0) = 0.0005208333333333333;
  sub_tet_int_proj_M[1](0, 1) = 0.0015625;
  sub_tet_int_proj_M[1](0, 2) = 0.00026041666666666666;
  sub_tet_int_proj_M[1](0, 3) = 0.00026041666666666666;
  sub_tet_int_proj_M[1](1, 0) = 0.0015625;
  sub_tet_int_proj_M[1](1, 1) = 0.008333333333333333;
  sub_tet_int_proj_M[1](1, 2) = 0.0015625;
  sub_tet_int_proj_M[1](1, 3) = 0.0015625;
  sub_tet_int_proj_M[1](2, 0) = 0.00026041666666666666;
  sub_tet_int_proj_M[1](2, 1) = 0.0015625;
  sub_tet_int_proj_M[1](2, 2) = 0.0005208333333333333;
  sub_tet_int_proj_M[1](2, 3) = 0.00026041666666666666;
  sub_tet_int_proj_M[1](3, 0) = 0.00026041666666666666;
  sub_tet_int_proj_M[1](3, 1) = 0.0015625;
  sub_tet_int_proj_M[1](3, 2) = 0.00026041666666666666;
  sub_tet_int_proj_M[1](3, 3) = 0.0005208333333333333;
  sub_tet_int_proj_M[2](0, 0) = 0.0005208333333333333;
  sub_tet_int_proj_M[2](0, 1) = 0.00026041666666666666;
  sub_tet_int_proj_M[2](0, 2) = 0.0015625;
  sub_tet_int_proj_M[2](0, 3) = 0.00026041666666666666;
  sub_tet_int_proj_M[2](1, 0) = 0.00026041666666666666;
  sub_tet_int_proj_M[2](1, 1) = 0.0005208333333333333;
  sub_tet_int_proj_M[2](1, 2) = 0.0015625;
  sub_tet_int_proj_M[2](1, 3) = 0.00026041666666666666;
  sub_tet_int_proj_M[2](2, 0) = 0.0015625;
  sub_tet_int_proj_M[2](2, 1) = 0.0015625;
  sub_tet_int_proj_M[2](2, 2) = 0.008333333333333333;
  sub_tet_int_proj_M[2](2, 3) = 0.0015625;
  sub_tet_int_proj_M[2](3, 0) = 0.00026041666666666666;
  sub_tet_int_proj_M[2](3, 1) = 0.00026041666666666666;
  sub_tet_int_proj_M[2](3, 2) = 0.0015625;
  sub_tet_int_proj_M[2](3, 3) = 0.0005208333333333333;
  sub_tet_int_proj_M[3](0, 0) = 0.0005208333333333333;
  sub_tet_int_proj_M[3](0, 1) = 0.00026041666666666666;
  sub_tet_int_proj_M[3](0, 2) = 0.00026041666666666666;
  sub_tet_int_proj_M[3](0, 3) = 0.0015625;
  sub_tet_int_proj_M[3](1, 0) = 0.00026041666666666666;
  sub_tet_int_proj_M[3](1, 1) = 0.0005208333333333333;
  sub_tet_int_proj_M[3](1, 2) = 0.00026041666666666666;
  sub_tet_int_proj_M[3](1, 3) = 0.0015625;
  sub_tet_int_proj_M[3](2, 0) = 0.00026041666666666666;
  sub_tet_int_proj_M[3](2, 1) = 0.00026041666666666666;
  sub_tet_int_proj_M[3](2, 2) = 0.0005208333333333333;
  sub_tet_int_proj_M[3](2, 3) = 0.0015625;
  sub_tet_int_proj_M[3](3, 0) = 0.0015625;
  sub_tet_int_proj_M[3](3, 1) = 0.0015625;
  sub_tet_int_proj_M[3](3, 2) = 0.0015625;
  sub_tet_int_proj_M[3](3, 3) = 0.008333333333333333;
  sub_tet_int_proj_M[4](0, 0) = 0.0004557291666666667;
  sub_tet_int_proj_M[4](0, 1) = 0.0008463541666666667;
  sub_tet_int_proj_M[4](0, 2) = 0.0003255208333333333;
  sub_tet_int_proj_M[4](0, 3) = 0.0003255208333333333;
  sub_tet_int_proj_M[4](1, 0) = 0.0008463541666666667;
  sub_tet_int_proj_M[4](1, 1) = 0.002018229166666667;
  sub_tet_int_proj_M[4](1, 2) = 0.0008463541666666667;
  sub_tet_int_proj_M[4](1, 3) = 0.0008463541666666667;
  sub_tet_int_proj_M[4](2, 0) = 0.0003255208333333333;
  sub_tet_int_proj_M[4](2, 1) = 0.0008463541666666667;
  sub_tet_int_proj_M[4](2, 2) = 0.0004557291666666667;
  sub_tet_int_proj_M[4](2, 3) = 0.0003255208333333333;
  sub_tet_int_proj_M[4](3, 0) = 0.0003255208333333333;
  sub_tet_int_proj_M[4](3, 1) = 0.0008463541666666667;
  sub_tet_int_proj_M[4](3, 2) = 0.0003255208333333333;
  sub_tet_int_proj_M[4](3, 3) = 0.0004557291666666667;
  sub_tet_int_proj_M[5](0, 0) = 0.00006510416666666667;
  sub_tet_int_proj_M[5](0, 1) = 0.0001953125;
  sub_tet_int_proj_M[5](0, 2) = 0.0001953125;
  sub_tet_int_proj_M[5](0, 3) = 0.0001953125;
  sub_tet_int_proj_M[5](1, 0) = 0.0001953125;
  sub_tet_int_proj_M[5](1, 1) = 0.0011067708333333333;
  sub_tet_int_proj_M[5](1, 2) = 0.0009765625;
  sub_tet_int_proj_M[5](1, 3) = 0.0009765625;
  sub_tet_int_proj_M[5](2, 0) = 0.0001953125;
  sub_tet_int_proj_M[5](2, 1) = 0.0009765625;
  sub_tet_int_proj_M[5](2, 2) = 0.0011067708333333333;
  sub_tet_int_proj_M[5](2, 3) = 0.0009765625;
  sub_tet_int_proj_M[5](3, 0) = 0.0001953125;
  sub_tet_int_proj_M[5](3, 1) = 0.0009765625;
  sub_tet_int_proj_M[5](3, 2) = 0.0009765625;
  sub_tet_int_proj_M[5](3, 3) = 0.0011067708333333333;
  sub_tet_int_proj_M[6](0, 0) = 0.0004557291666666667;
  sub_tet_int_proj_M[6](0, 1) = 0.0003255208333333333;
  sub_tet_int_proj_M[6](0, 2) = 0.0003255208333333333;
  sub_tet_int_proj_M[6](0, 3) = 0.0008463541666666667;
  sub_tet_int_proj_M[6](1, 0) = 0.0003255208333333333;
  sub_tet_int_proj_M[6](1, 1) = 0.0004557291666666667;
  sub_tet_int_proj_M[6](1, 2) = 0.0003255208333333333;
  sub_tet_int_proj_M[6](1, 3) = 0.0008463541666666667;
  sub_tet_int_proj_M[6](2, 0) = 0.0003255208333333333;
  sub_tet_int_proj_M[6](2, 1) = 0.0003255208333333333;
  sub_tet_int_proj_M[6](2, 2) = 0.0004557291666666667;
  sub_tet_int_proj_M[6](2, 3) = 0.0008463541666666667;
  sub_tet_int_proj_M[6](3, 0) = 0.0008463541666666667;
  sub_tet_int_proj_M[6](3, 1) = 0.0008463541666666667;
  sub_tet_int_proj_M[6](3, 2) = 0.0008463541666666667;
  sub_tet_int_proj_M[6](3, 3) = 0.002018229166666667;
  sub_tet_int_proj_M[7](0, 0) = 0.0011067708333333333;
  sub_tet_int_proj_M[7](0, 1) = 0.0009765625;
  sub_tet_int_proj_M[7](0, 2) = 0.0001953125;
  sub_tet_int_proj_M[7](0, 3) = 0.0009765625;
  sub_tet_int_proj_M[7](1, 0) = 0.0009765625;
  sub_tet_int_proj_M[7](1, 1) = 0.0011067708333333333;
  sub_tet_int_proj_M[7](1, 2) = 0.0001953125;
  sub_tet_int_proj_M[7](1, 3) = 0.0009765625;
  sub_tet_int_proj_M[7](2, 0) = 0.0001953125;
  sub_tet_int_proj_M[7](2, 1) = 0.0001953125;
  sub_tet_int_proj_M[7](2, 2) = 0.00006510416666666667;
  sub_tet_int_proj_M[7](2, 3) = 0.0001953125;
  sub_tet_int_proj_M[7](3, 0) = 0.0009765625;
  sub_tet_int_proj_M[7](3, 1) = 0.0009765625;
  sub_tet_int_proj_M[7](3, 2) = 0.0001953125;
  sub_tet_int_proj_M[7](3, 3) = 0.0011067708333333333;
  sub_tet_int_proj_M[8](0, 0) = 0.0011067708333333333;
  sub_tet_int_proj_M[8](0, 1) = 0.0009765625;
  sub_tet_int_proj_M[8](0, 2) = 0.0009765625;
  sub_tet_int_proj_M[8](0, 3) = 0.0001953125;
  sub_tet_int_proj_M[8](1, 0) = 0.0009765625;
  sub_tet_int_proj_M[8](1, 1) = 0.0011067708333333333;
  sub_tet_int_proj_M[8](1, 2) = 0.0009765625;
  sub_tet_int_proj_M[8](1, 3) = 0.0001953125;
  sub_tet_int_proj_M[8](2, 0) = 0.0009765625;
  sub_tet_int_proj_M[8](2, 1) = 0.0009765625;
  sub_tet_int_proj_M[8](2, 2) = 0.0011067708333333333;
  sub_tet_int_proj_M[8](2, 3) = 0.0001953125;
  sub_tet_int_proj_M[8](3, 0) = 0.0001953125;
  sub_tet_int_proj_M[8](3, 1) = 0.0001953125;
  sub_tet_int_proj_M[8](3, 2) = 0.0001953125;
  sub_tet_int_proj_M[8](3, 3) = 0.00006510416666666667;
  sub_tet_int_proj_M[9](0, 0) = 0.0004557291666666667;
  sub_tet_int_proj_M[9](0, 1) = 0.0003255208333333333;
  sub_tet_int_proj_M[9](0, 2) = 0.0008463541666666667;
  sub_tet_int_proj_M[9](0, 3) = 0.0003255208333333333;
  sub_tet_int_proj_M[9](1, 0) = 0.0003255208333333333;
  sub_tet_int_proj_M[9](1, 1) = 0.0004557291666666667;
  sub_tet_int_proj_M[9](1, 2) = 0.0008463541666666667;
  sub_tet_int_proj_M[9](1, 3) = 0.0003255208333333333;
  sub_tet_int_proj_M[9](2, 0) = 0.0008463541666666667;
  sub_tet_int_proj_M[9](2, 1) = 0.0008463541666666667;
  sub_tet_int_proj_M[9](2, 2) = 0.002018229166666667;
  sub_tet_int_proj_M[9](2, 3) = 0.0008463541666666667;
  sub_tet_int_proj_M[9](3, 0) = 0.0003255208333333333;
  sub_tet_int_proj_M[9](3, 1) = 0.0003255208333333333;
  sub_tet_int_proj_M[9](3, 2) = 0.0008463541666666667;
  sub_tet_int_proj_M[9](3, 3) = 0.0004557291666666667;
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
  return sub_tet_int_proj_M;
}

HPC_HOST_DEVICE inline subtet_int_t get_subtet_int() noexcept {
  subtet_int_t subtet_int;
  assert(subtet_int.size() == 12);
  assert(subtet_int[0].size() == 4);
  subtet_int[0][0] = 0.013020833333333334;
  subtet_int[0][1] = 0.0026041666666666665;
  subtet_int[0][2] = 0.0026041666666666665;
  subtet_int[0][3] = 0.0026041666666666665;
  subtet_int[1][0] = 0.0026041666666666665;
  subtet_int[1][1] = 0.013020833333333334;
  subtet_int[1][2] = 0.0026041666666666665;
  subtet_int[1][3] = 0.0026041666666666665;
  subtet_int[2][0] = 0.0026041666666666665;
  subtet_int[2][1] = 0.0026041666666666665;
  subtet_int[2][2] = 0.013020833333333334;
  subtet_int[2][3] = 0.0026041666666666665;
  subtet_int[3][0] = 0.0026041666666666665;
  subtet_int[3][1] = 0.0026041666666666665;
  subtet_int[3][2] = 0.0026041666666666665;
  subtet_int[3][3] = 0.013020833333333334;
  subtet_int[4][0] = 0.001953125;
  subtet_int[4][1] = 0.004557291666666667;
  subtet_int[4][2] = 0.001953125;
  subtet_int[4][3] = 0.001953125;
  subtet_int[5][0] = 0.0006510416666666666;
  subtet_int[5][1] = 0.0032552083333333335;
  subtet_int[5][2] = 0.0032552083333333335;
  subtet_int[5][3] = 0.0032552083333333335;
  subtet_int[6][0] = 0.001953125;
  subtet_int[6][1] = 0.001953125;
  subtet_int[6][2] = 0.001953125;
  subtet_int[6][3] = 0.004557291666666667;
  subtet_int[7][0] = 0.0032552083333333335;
  subtet_int[7][1] = 0.0032552083333333335;
  subtet_int[7][2] = 0.0006510416666666666;
  subtet_int[7][3] = 0.0032552083333333335;
  subtet_int[8][0] = 0.0032552083333333335;
  subtet_int[8][1] = 0.0032552083333333335;
  subtet_int[8][2] = 0.0032552083333333335;
  subtet_int[8][3] = 0.0006510416666666666;
  subtet_int[9][0] = 0.001953125;
  subtet_int[9][1] = 0.001953125;
  subtet_int[9][2] = 0.004557291666666667;
  subtet_int[9][3] = 0.001953125;
  subtet_int[10][0] = 0.0032552083333333335;
  subtet_int[10][1] = 0.0006510416666666666;
  subtet_int[10][2] = 0.0032552083333333335;
  subtet_int[10][3] = 0.0032552083333333335;
  subtet_int[11][0] = 0.004557291666666667;
  subtet_int[11][1] = 0.001953125;
  subtet_int[11][2] = 0.001953125;
  subtet_int[11][3] = 0.001953125;
  return subtet_int;
}

HPC_HOST_DEVICE constexpr inline matrix4x4<double> get_parent_M_inv() noexcept {
  return matrix4x4<double>(
      96.0,-24.0,-24.0,-24.0,
     -24.0, 96.0,-24.0,-24.0,
     -24.0,-24.0, 96.0,-24.0,
     -24.0,-24.0,-24.0, 96.0);
}

HPC_HOST_DEVICE inline O_t get_O_inv(O_t const O) noexcept {
  O_t O_inv;
  for (int tet = 0; tet < 12; ++tet) {
    O_inv[tet] = inverse(O[tet]);
  }
  return O_inv;
}

HPC_HOST_DEVICE inline matrix4x4<double> get_M_inv(hpc::array<double, 12> const O_det) noexcept {
  auto M = matrix4x4<double>::zero();
  auto const sub_tet_int_proj_M = get_subtet_proj_M();
  for (int tet = 0; tet < 12; ++tet) {
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 4; ++j) {
        M(i, j) += O_det[tet] * sub_tet_int_proj_M[tet](i, j);
      }
    }
  }
  return inverse(M);
}

HPC_HOST_DEVICE inline SOL_t get_SOL(hpc::array<double, 12> O_det, O_t O_inv,
    subtet_int_t subtet_int, S_t S) noexcept {
  SOL_t SOL;
  for (auto& a : SOL)
  for (auto& b : a)
    b = hpc::vector3<double>::zero();
  for (int tet = 0; tet < 12; ++tet) {
    for (int node = 0; node < 10; ++node) {
      for (int dim1 = 0; dim1 < 3; ++dim1) {
        for (int dim2 = 0; dim2 < 3; ++dim2) {
          for (int pt = 0; pt < 4; ++pt) {
            SOL[pt][node](dim1) += O_det[tet] * S[tet][node](dim2) *
                                   O_inv[tet](dim2, dim1) *
                                   subtet_int[tet][pt];
          }
        }
      }
    }
  }
  return SOL;
}

HPC_HOST_DEVICE inline hpc::array<double, 4> get_DOL(
    hpc::array<double, 12> O_det, subtet_int_t subtet_int) noexcept {
  hpc::array<double, 4> DOL;
  for (auto& a : DOL) a = 0.0;
  for (int tet = 0; tet < 12; ++tet) {
    for (int pt = 0; pt < 4; ++pt) {
      DOL[pt] += O_det[tet] * subtet_int[tet][pt];
    }
  }
  return DOL;
}

HPC_HOST_DEVICE inline hpc::array<hpc::vector3<double>, 4> get_ref_points() noexcept {
  hpc::array<hpc::vector3<double>, 4> pts;
  pts[0](0) = 0.1381966011250105151795413165634361882280;
  pts[0](1) = 0.1381966011250105151795413165634361882280;
  pts[0](2) = 0.1381966011250105151795413165634361882280;
  pts[1](0) = 0.5854101966249684544613760503096914353161;
  pts[1](1) = 0.1381966011250105151795413165634361882280;
  pts[1](2) = 0.1381966011250105151795413165634361882280;
  pts[2](0) = 0.1381966011250105151795413165634361882280;
  pts[2](1) = 0.5854101966249684544613760503096914353161;
  pts[2](2) = 0.1381966011250105151795413165634361882280;
  pts[3](0) = 0.1381966011250105151795413165634361882280;
  pts[3](1) = 0.1381966011250105151795413165634361882280;
  pts[3](2) = 0.5854101966249684544613760503096914353161;
  return pts;
}

HPC_HOST_DEVICE inline hpc::array<double, 4> get_barycentric(hpc::vector3<double> const x) noexcept {
  hpc::array<double, 4> xi;
  xi[0] = 1.0 - x(0) - x(1) - x(2);
  xi[1] = x(0);
  xi[2] = x(1);
  xi[3] = x(2);
  return xi;
}

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

HPC_HOST_DEVICE inline double get_length(hpc::array<hpc::vector3<double>, 10> in) noexcept {
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

HPC_HOST_DEVICE inline hpc::array<hpc::array<hpc::vector3<double>, 10>, 4> get_basis_gradients(
    hpc::array<hpc::vector3<double>, 10> const node_coords) noexcept
{
  hpc::array<hpc::array<hpc::vector3<double>, 10>, 4> grad_N;
  for (auto& a : grad_N) {
    for (auto& b : a) {
      b = hpc::vector3<double>::zero();
    }
  }
  auto const ref_points = get_ref_points();
  auto const subtet_int = get_subtet_int();
  auto const S = get_S();
  auto const O = get_O(node_coords, S);
  auto const O_inv = get_O_inv(O);
  auto const O_det = get_O_det(O);
  auto const M_inv = get_M_inv(O_det);
  auto const SOL = get_SOL(O_det, O_inv, subtet_int, S);
  for (int node = 0; node < 10; ++node) {
    for (int pt = 0; pt < 4; ++pt) {
      auto const lambda = get_barycentric(ref_points[pt]);
      for (int d = 0; d < 3; ++d) {
        for (int l1 = 0; l1 < 4; ++l1) {
          for (int l2 = 0; l2 < 4; ++l2) {
            grad_N[pt][node](d) +=
                lambda[l1] * M_inv(l1, l2) * SOL[l2][node](d);
          }
        }
      }
    }
  }
  return grad_N;
}

HPC_HOST_DEVICE inline hpc::array<double, 4> get_volumes(
    hpc::array<hpc::vector3<double>, 10> const node_coords) noexcept
{
  // compute the projected |J| times integration weights
  constexpr double ip_weight = 1.0 / 24.0;
  hpc::array<double, 4> volumes;
  for (auto& a : volumes) a = 0.0;
  auto const ref_points = get_ref_points();
  auto const sub_tet_int = get_subtet_int();
  auto const S = get_S();
  auto const O = get_O(node_coords, S);
  auto const O_det = get_O_det(O);
  auto const DOL = get_DOL(O_det, sub_tet_int);
  auto const parent_M_inv = get_parent_M_inv();
  for (int pt = 0; pt < 4; ++pt) {
    auto const lambda = get_barycentric(ref_points[pt]);
    for (int l1 = 0; l1 < 4; ++l1) {
      for (int l2 = 0; l2 < 4; ++l2) {
        volumes[pt] +=
          lambda[l1] * parent_M_inv(l1, l2) * DOL[l2];
      }
    }
    volumes[pt] *= ip_weight;
  }
  return volumes;
}

} // namespace composite_tetrahedron

void initialize_composite_tetrahedron_V(state& s)
{
  auto const element_nodes_to_nodes = s.elements_to_nodes.cbegin();
  auto const nodes_to_x = s.x.cbegin();
  auto const points_to_V = s.V.begin();
  auto const elements_to_element_nodes = s.elements * s.nodes_in_element;
  auto const elements_to_points = s.elements * s.points_in_element;
  auto const nodes_in_element = s.nodes_in_element;
  auto const points_in_element = s.points_in_element;
  auto functor = [=] HPC_DEVICE (element_index const element) {
    auto const element_nodes = elements_to_element_nodes[element];
    hpc::array<hpc::vector3<double>, 10> node_coords;
    for (auto const node_in_element : nodes_in_element) {
      auto const node = element_nodes_to_nodes[element_nodes[node_in_element]];
      node_coords[node_in_element.get()] = nodes_to_x[node].load();
    }
    auto const volumes = composite_tetrahedron::get_volumes(node_coords);
#ifndef NDEBUG
    for (auto const volume : volumes) {
      assert(volume > 0.0);
    }
#endif
    auto const element_points = elements_to_points[element];
    for (auto const qp : points_in_element) {
      points_to_V[element_points[qp]] = volumes[qp.get()];
    }
  };
  hpc::for_each(hpc::device_policy(), s.elements, functor);
}

void initialize_composite_tetrahedron_grad_N(state& s) {
  auto const element_nodes_to_nodes = s.elements_to_nodes.cbegin();
  auto const nodes_to_x = s.x.cbegin();
  auto const point_nodes_to_grad_N = s.grad_N.begin();
  auto const elements_to_element_nodes = s.elements * s.nodes_in_element;
  auto const elements_to_points = s.elements * s.points_in_element;
  auto const points_to_point_nodes = s.points * s.nodes_in_element;
  auto const nodes_in_element = s.nodes_in_element;
  auto const points_in_element = s.points_in_element;
  auto functor = [=] HPC_DEVICE (element_index const element) {
    auto const element_nodes = elements_to_element_nodes[element];
    hpc::array<hpc::vector3<double>, 10> node_coords;
    for (auto const node_in_element : nodes_in_element) {
      auto const node = element_nodes_to_nodes[element_nodes[node_in_element]];
      node_coords[node_in_element.get()] = nodes_to_x[node].load();
    }
    auto const grad_N = composite_tetrahedron::get_basis_gradients(node_coords);
    auto const element_points = elements_to_points[element];
    for (auto const qp : points_in_element) {
      auto const point = element_points[qp];
      auto const point_nodes = points_to_point_nodes[point];
      for (auto const a : nodes_in_element) {
        auto const point_node = point_nodes[a];
        point_nodes_to_grad_N[point_node] = grad_N[qp.get()][a.get()];
      }
    }
  };
  hpc::for_each(hpc::device_policy(), s.elements, functor);
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
