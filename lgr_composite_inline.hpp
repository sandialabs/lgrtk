#pragma once

#include <hpc_array.hpp>
#include <hpc_vector3.hpp>
#include <lgr_vector4.hpp>
#include <hpc_matrix3x3.hpp>
#include <lgr_matrix4x4.hpp>

namespace lgr {
namespace composite_tetrahedron {

using S_t = hpc::array<hpc::array<hpc::vector3<double>, 10>, 12>;
using gamma_t = hpc::array<hpc::array<hpc::array<double, 10>, 10>, 12>;
using subtet_proj_t = hpc::array<matrix4x4<double>, 12>;
using subtet_int_t = hpc::array<hpc::array<double, 4>, 12>;
using O_t = hpc::array<hpc::matrix3x3<double>, 12>;
using SOL_t = hpc::array<hpc::array<hpc::vector3<double>, 10>, 4>;

HPC_NOINLINE HPC_HOST_DEVICE inline void get_S(S_t& S) noexcept {
  for (auto& a : S) {
    for (auto& b : a) {
      b = hpc::vector3<double>::zero();
    }
  }
  S[0][0](0) = -2;
  S[0][0](1) = -2;
  S[0][0](2) = -2;
  S[0][4](0) = 2;
  S[0][6](1) = 2;
  S[0][7](2) = 2;
  S[1][1](0) = 2;
  S[1][4](0) = -2;
  S[1][4](1) = -2;
  S[1][4](2) = -2;
  S[1][5](1) = 2;
  S[1][8](2) = 2;
  S[2][2](1) = 2;
  S[2][5](0) = 2;
  S[2][6](0) = -2;
  S[2][6](1) = -2;
  S[2][6](2) = -2;
  S[2][9](2) = 2;
  S[3][3](2) = 2;
  S[3][7](0) = -2;
  S[3][7](1) = -2;
  S[3][7](2) = -2;
  S[3][8](0) = 2;
  S[3][9](1) = 2;
  S[4][4](0) = -0.6666666666666666;
  S[4][4](1) = -2;
  S[4][4](2) = -2;
  S[4][5](0) = 1.3333333333333333;
  S[4][5](1) = 2;
  S[4][6](0) = -0.6666666666666666;
  S[4][7](0) = -0.6666666666666666;
  S[4][8](0) = 1.3333333333333333;
  S[4][8](2) = 2;
  S[4][9](0) = -0.6666666666666666;
  S[5][4](0) = -0.6666666666666666;
  S[5][4](1) = -0.6666666666666666;
  S[5][4](2) = -0.6666666666666666;
  S[5][5](0) = 1.3333333333333333;
  S[5][5](1) = 1.3333333333333333;
  S[5][5](2) = -0.6666666666666666;
  S[5][6](0) = -0.6666666666666666;
  S[5][6](1) = -0.6666666666666666;
  S[5][6](2) = -0.6666666666666666;
  S[5][7](0) = -0.6666666666666666;
  S[5][7](1) = -0.6666666666666666;
  S[5][7](2) = -0.6666666666666666;
  S[5][8](0) = 1.3333333333333333;
  S[5][8](1) = -0.6666666666666666;
  S[5][8](2) = 1.3333333333333333;
  S[5][9](0) = -0.6666666666666666;
  S[5][9](1) = 1.3333333333333333;
  S[5][9](2) = 1.3333333333333333;
  S[6][4](2) = -0.6666666666666666;
  S[6][5](2) = -0.6666666666666666;
  S[6][6](2) = -0.6666666666666666;
  S[6][7](0) = -2;
  S[6][7](1) = -2;
  S[6][7](2) = -0.6666666666666666;
  S[6][8](0) = 2;
  S[6][8](2) = 1.3333333333333333;
  S[6][9](1) = 2;
  S[6][9](2) = 1.3333333333333333;
  S[7][4](1) = -1.3333333333333333;
  S[7][4](2) = -2;
  S[7][5](1) = 0.6666666666666666;
  S[7][6](1) = 0.6666666666666666;
  S[7][7](0) = -2;
  S[7][7](1) = -1.3333333333333333;
  S[7][8](0) = 2;
  S[7][8](1) = 0.6666666666666666;
  S[7][8](2) = 2;
  S[7][9](1) = 0.6666666666666666;
  S[8][4](1) = -2;
  S[8][4](2) = -1.3333333333333333;
  S[8][5](0) = 2;
  S[8][5](1) = 2;
  S[8][5](2) = 0.6666666666666666;
  S[8][6](0) = -2;
  S[8][6](2) = -1.3333333333333333;
  S[8][7](2) = 0.6666666666666666;
  S[8][8](2) = 0.6666666666666666;
  S[8][9](2) = 0.6666666666666666;
  S[9][4](1) = -0.6666666666666666;
  S[9][5](0) = 2;
  S[9][5](1) = 1.3333333333333333;
  S[9][6](0) = -2;
  S[9][6](1) = -0.6666666666666666;
  S[9][6](2) = -2;
  S[9][7](1) = -0.6666666666666666;
  S[9][8](1) = -0.6666666666666666;
  S[9][9](1) = 1.3333333333333333;
  S[9][9](2) = 2;
  S[10][4](0) = 0.6666666666666666;
  S[10][5](0) = 0.6666666666666666;
  S[10][6](0) = -1.3333333333333333;
  S[10][6](2) = -2;
  S[10][7](0) = -1.3333333333333333;
  S[10][7](1) = -2;
  S[10][8](0) = 0.6666666666666666;
  S[10][9](0) = 0.6666666666666666;
  S[10][9](1) = 2;
  S[10][9](2) = 2;
  S[11][4](0) = 0.6666666666666666;
  S[11][4](1) = -1.3333333333333333;
  S[11][4](2) = -1.3333333333333333;
  S[11][5](0) = 0.6666666666666666;
  S[11][5](1) = 0.6666666666666666;
  S[11][5](2) = 0.6666666666666666;
  S[11][6](0) = -1.3333333333333333;
  S[11][6](1) = 0.6666666666666666;
  S[11][6](2) = -1.3333333333333333;
  S[11][7](0) = -1.3333333333333333;
  S[11][7](1) = -1.3333333333333333;
  S[11][7](2) = 0.6666666666666666;
  S[11][8](0) = 0.6666666666666666;
  S[11][8](1) = 0.6666666666666666;
  S[11][8](2) = 0.6666666666666666;
  S[11][9](0) = 0.6666666666666666;
  S[11][9](1) = 0.6666666666666666;
  S[11][9](2) = 0.6666666666666666;
}

HPC_NOINLINE HPC_HOST_DEVICE inline void get_gamma(gamma_t& gamma) noexcept {
  for (auto& a : gamma) {
    for (auto& b : a) {
      for (auto& c : b) {
        c = 0.0;
      }
    }
  }
  gamma[0][0][0] = 0.0020833333333333333;
  gamma[0][0][4] = 0.0010416666666666667;
  gamma[0][0][6] = 0.0010416666666666667;
  gamma[0][0][7] = 0.0010416666666666667;
  gamma[0][4][0] = 0.0010416666666666667;
  gamma[0][4][4] = 0.0020833333333333333;
  gamma[0][4][6] = 0.0010416666666666667;
  gamma[0][4][7] = 0.0010416666666666667;
  gamma[0][6][0] = 0.0010416666666666667;
  gamma[0][6][4] = 0.0010416666666666667;
  gamma[0][6][6] = 0.0020833333333333333;
  gamma[0][6][7] = 0.0010416666666666667;
  gamma[0][7][0] = 0.0010416666666666667;
  gamma[0][7][4] = 0.0010416666666666667;
  gamma[0][7][6] = 0.0010416666666666667;
  gamma[0][7][7] = 0.0020833333333333333;
  gamma[1][1][1] = 0.0020833333333333333;
  gamma[1][1][4] = 0.0010416666666666667;
  gamma[1][1][5] = 0.0010416666666666667;
  gamma[1][1][8] = 0.0010416666666666667;
  gamma[1][4][1] = 0.0010416666666666667;
  gamma[1][4][4] = 0.0020833333333333333;
  gamma[1][4][5] = 0.0010416666666666667;
  gamma[1][4][8] = 0.0010416666666666667;
  gamma[1][5][1] = 0.0010416666666666667;
  gamma[1][5][4] = 0.0010416666666666667;
  gamma[1][5][5] = 0.0020833333333333333;
  gamma[1][5][8] = 0.0010416666666666667;
  gamma[1][8][1] = 0.0010416666666666667;
  gamma[1][8][4] = 0.0010416666666666667;
  gamma[1][8][5] = 0.0010416666666666667;
  gamma[1][8][8] = 0.0020833333333333333;
  gamma[2][2][2] = 0.0020833333333333333;
  gamma[2][2][5] = 0.0010416666666666667;
  gamma[2][2][6] = 0.0010416666666666667;
  gamma[2][2][9] = 0.0010416666666666667;
  gamma[2][5][2] = 0.0010416666666666667;
  gamma[2][5][5] = 0.0020833333333333333;
  gamma[2][5][6] = 0.0010416666666666667;
  gamma[2][5][9] = 0.0010416666666666667;
  gamma[2][6][2] = 0.0010416666666666667;
  gamma[2][6][5] = 0.0010416666666666667;
  gamma[2][6][6] = 0.0020833333333333333;
  gamma[2][6][9] = 0.0010416666666666667;
  gamma[2][9][2] = 0.0010416666666666667;
  gamma[2][9][5] = 0.0010416666666666667;
  gamma[2][9][6] = 0.0010416666666666667;
  gamma[2][9][9] = 0.0020833333333333333;
  gamma[3][3][3] = 0.0020833333333333333;
  gamma[3][3][7] = 0.0010416666666666667;
  gamma[3][3][8] = 0.0010416666666666667;
  gamma[3][3][9] = 0.0010416666666666667;
  gamma[3][7][3] = 0.0010416666666666667;
  gamma[3][7][7] = 0.0020833333333333333;
  gamma[3][7][8] = 0.0010416666666666667;
  gamma[3][7][9] = 0.0010416666666666667;
  gamma[3][8][3] = 0.0010416666666666667;
  gamma[3][8][7] = 0.0010416666666666667;
  gamma[3][8][8] = 0.0020833333333333333;
  gamma[3][8][9] = 0.0010416666666666667;
  gamma[3][9][3] = 0.0010416666666666667;
  gamma[3][9][7] = 0.0010416666666666667;
  gamma[3][9][8] = 0.0010416666666666667;
  gamma[3][9][9] = 0.0020833333333333333;
  gamma[4][4][4] = 0.001244212962962963;
  gamma[4][4][5] = 0.0007233796296296296;
  gamma[4][4][6] = 0.00011574074074074075;
  gamma[4][4][7] = 0.00011574074074074075;
  gamma[4][4][8] = 0.0007233796296296296;
  gamma[4][4][9] = 0.00011574074074074075;
  gamma[4][5][4] = 0.0007233796296296296;
  gamma[4][5][5] = 0.001244212962962963;
  gamma[4][5][6] = 0.00011574074074074075;
  gamma[4][5][7] = 0.00011574074074074075;
  gamma[4][5][8] = 0.0007233796296296296;
  gamma[4][5][9] = 0.00011574074074074075;
  gamma[4][6][4] = 0.00011574074074074075;
  gamma[4][6][5] = 0.00011574074074074075;
  gamma[4][6][6] = 0.000028935185185185186;
  gamma[4][6][7] = 0.000028935185185185186;
  gamma[4][6][8] = 0.00011574074074074075;
  gamma[4][6][9] = 0.000028935185185185186;
  gamma[4][7][4] = 0.00011574074074074075;
  gamma[4][7][5] = 0.00011574074074074075;
  gamma[4][7][6] = 0.000028935185185185186;
  gamma[4][7][7] = 0.000028935185185185186;
  gamma[4][7][8] = 0.00011574074074074075;
  gamma[4][7][9] = 0.000028935185185185186;
  gamma[4][8][4] = 0.0007233796296296296;
  gamma[4][8][5] = 0.0007233796296296296;
  gamma[4][8][6] = 0.00011574074074074075;
  gamma[4][8][7] = 0.00011574074074074075;
  gamma[4][8][8] = 0.001244212962962963;
  gamma[4][8][9] = 0.00011574074074074075;
  gamma[4][9][4] = 0.00011574074074074075;
  gamma[4][9][5] = 0.00011574074074074075;
  gamma[4][9][6] = 0.000028935185185185186;
  gamma[4][9][7] = 0.000028935185185185186;
  gamma[4][9][8] = 0.00011574074074074075;
  gamma[4][9][9] = 0.000028935185185185186;
  gamma[5][4][4] = 0.000028935185185185186;
  gamma[5][4][5] = 0.00011574074074074075;
  gamma[5][4][6] = 0.000028935185185185186;
  gamma[5][4][7] = 0.000028935185185185186;
  gamma[5][4][8] = 0.00011574074074074075;
  gamma[5][4][9] = 0.00011574074074074075;
  gamma[5][5][4] = 0.00011574074074074075;
  gamma[5][5][5] = 0.001244212962962963;
  gamma[5][5][6] = 0.00011574074074074075;
  gamma[5][5][7] = 0.00011574074074074075;
  gamma[5][5][8] = 0.0007233796296296296;
  gamma[5][5][9] = 0.0007233796296296296;
  gamma[5][6][4] = 0.000028935185185185186;
  gamma[5][6][5] = 0.00011574074074074075;
  gamma[5][6][6] = 0.000028935185185185186;
  gamma[5][6][7] = 0.000028935185185185186;
  gamma[5][6][8] = 0.00011574074074074075;
  gamma[5][6][9] = 0.00011574074074074075;
  gamma[5][7][4] = 0.000028935185185185186;
  gamma[5][7][5] = 0.00011574074074074075;
  gamma[5][7][6] = 0.000028935185185185186;
  gamma[5][7][7] = 0.000028935185185185186;
  gamma[5][7][8] = 0.00011574074074074075;
  gamma[5][7][9] = 0.00011574074074074075;
  gamma[5][8][4] = 0.00011574074074074075;
  gamma[5][8][5] = 0.0007233796296296296;
  gamma[5][8][6] = 0.00011574074074074075;
  gamma[5][8][7] = 0.00011574074074074075;
  gamma[5][8][8] = 0.001244212962962963;
  gamma[5][8][9] = 0.0007233796296296296;
  gamma[5][9][4] = 0.00011574074074074075;
  gamma[5][9][5] = 0.0007233796296296296;
  gamma[5][9][6] = 0.00011574074074074075;
  gamma[5][9][7] = 0.00011574074074074075;
  gamma[5][9][8] = 0.0007233796296296296;
  gamma[5][9][9] = 0.001244212962962963;
  gamma[6][4][4] = 0.000028935185185185186;
  gamma[6][4][5] = 0.000028935185185185186;
  gamma[6][4][6] = 0.000028935185185185186;
  gamma[6][4][7] = 0.00011574074074074075;
  gamma[6][4][8] = 0.00011574074074074075;
  gamma[6][4][9] = 0.00011574074074074075;
  gamma[6][5][4] = 0.000028935185185185186;
  gamma[6][5][5] = 0.000028935185185185186;
  gamma[6][5][6] = 0.000028935185185185186;
  gamma[6][5][7] = 0.00011574074074074075;
  gamma[6][5][8] = 0.00011574074074074075;
  gamma[6][5][9] = 0.00011574074074074075;
  gamma[6][6][4] = 0.000028935185185185186;
  gamma[6][6][5] = 0.000028935185185185186;
  gamma[6][6][6] = 0.000028935185185185186;
  gamma[6][6][7] = 0.00011574074074074075;
  gamma[6][6][8] = 0.00011574074074074075;
  gamma[6][6][9] = 0.00011574074074074075;
  gamma[6][7][4] = 0.00011574074074074075;
  gamma[6][7][5] = 0.00011574074074074075;
  gamma[6][7][6] = 0.00011574074074074075;
  gamma[6][7][7] = 0.001244212962962963;
  gamma[6][7][8] = 0.0007233796296296296;
  gamma[6][7][9] = 0.0007233796296296296;
  gamma[6][8][4] = 0.00011574074074074075;
  gamma[6][8][5] = 0.00011574074074074075;
  gamma[6][8][6] = 0.00011574074074074075;
  gamma[6][8][7] = 0.0007233796296296296;
  gamma[6][8][8] = 0.001244212962962963;
  gamma[6][8][9] = 0.0007233796296296296;
  gamma[6][9][4] = 0.00011574074074074075;
  gamma[6][9][5] = 0.00011574074074074075;
  gamma[6][9][6] = 0.00011574074074074075;
  gamma[6][9][7] = 0.0007233796296296296;
  gamma[6][9][8] = 0.0007233796296296296;
  gamma[6][9][9] = 0.001244212962962963;
  gamma[7][4][4] = 0.001244212962962963;
  gamma[7][4][5] = 0.00011574074074074075;
  gamma[7][4][6] = 0.00011574074074074075;
  gamma[7][4][7] = 0.0007233796296296296;
  gamma[7][4][8] = 0.0007233796296296296;
  gamma[7][4][9] = 0.00011574074074074075;
  gamma[7][5][4] = 0.00011574074074074075;
  gamma[7][5][5] = 0.000028935185185185186;
  gamma[7][5][6] = 0.000028935185185185186;
  gamma[7][5][7] = 0.00011574074074074075;
  gamma[7][5][8] = 0.00011574074074074075;
  gamma[7][5][9] = 0.000028935185185185186;
  gamma[7][6][4] = 0.00011574074074074075;
  gamma[7][6][5] = 0.000028935185185185186;
  gamma[7][6][6] = 0.000028935185185185186;
  gamma[7][6][7] = 0.00011574074074074075;
  gamma[7][6][8] = 0.00011574074074074075;
  gamma[7][6][9] = 0.000028935185185185186;
  gamma[7][7][4] = 0.0007233796296296296;
  gamma[7][7][5] = 0.00011574074074074075;
  gamma[7][7][6] = 0.00011574074074074075;
  gamma[7][7][7] = 0.001244212962962963;
  gamma[7][7][8] = 0.0007233796296296296;
  gamma[7][7][9] = 0.00011574074074074075;
  gamma[7][8][4] = 0.0007233796296296296;
  gamma[7][8][5] = 0.00011574074074074075;
  gamma[7][8][6] = 0.00011574074074074075;
  gamma[7][8][7] = 0.0007233796296296296;
  gamma[7][8][8] = 0.001244212962962963;
  gamma[7][8][9] = 0.00011574074074074075;
  gamma[7][9][4] = 0.00011574074074074075;
  gamma[7][9][5] = 0.000028935185185185186;
  gamma[7][9][6] = 0.000028935185185185186;
  gamma[7][9][7] = 0.00011574074074074075;
  gamma[7][9][8] = 0.00011574074074074075;
  gamma[7][9][9] = 0.000028935185185185186;
  gamma[8][4][4] = 0.001244212962962963;
  gamma[8][4][5] = 0.0007233796296296296;
  gamma[8][4][6] = 0.0007233796296296296;
  gamma[8][4][7] = 0.00011574074074074075;
  gamma[8][4][8] = 0.00011574074074074075;
  gamma[8][4][9] = 0.00011574074074074075;
  gamma[8][5][4] = 0.0007233796296296296;
  gamma[8][5][5] = 0.001244212962962963;
  gamma[8][5][6] = 0.0007233796296296296;
  gamma[8][5][7] = 0.00011574074074074075;
  gamma[8][5][8] = 0.00011574074074074075;
  gamma[8][5][9] = 0.00011574074074074075;
  gamma[8][6][4] = 0.0007233796296296296;
  gamma[8][6][5] = 0.0007233796296296296;
  gamma[8][6][6] = 0.001244212962962963;
  gamma[8][6][7] = 0.00011574074074074075;
  gamma[8][6][8] = 0.00011574074074074075;
  gamma[8][6][9] = 0.00011574074074074075;
  gamma[8][7][4] = 0.00011574074074074075;
  gamma[8][7][5] = 0.00011574074074074075;
  gamma[8][7][6] = 0.00011574074074074075;
  gamma[8][7][7] = 0.000028935185185185186;
  gamma[8][7][8] = 0.000028935185185185186;
  gamma[8][7][9] = 0.000028935185185185186;
  gamma[8][8][4] = 0.00011574074074074075;
  gamma[8][8][5] = 0.00011574074074074075;
  gamma[8][8][6] = 0.00011574074074074075;
  gamma[8][8][7] = 0.000028935185185185186;
  gamma[8][8][8] = 0.000028935185185185186;
  gamma[8][8][9] = 0.000028935185185185186;
  gamma[8][9][4] = 0.00011574074074074075;
  gamma[8][9][5] = 0.00011574074074074075;
  gamma[8][9][6] = 0.00011574074074074075;
  gamma[8][9][7] = 0.000028935185185185186;
  gamma[8][9][8] = 0.000028935185185185186;
  gamma[8][9][9] = 0.000028935185185185186;
  gamma[9][4][4] = 0.000028935185185185186;
  gamma[9][4][5] = 0.00011574074074074075;
  gamma[9][4][6] = 0.00011574074074074075;
  gamma[9][4][7] = 0.000028935185185185186;
  gamma[9][4][8] = 0.000028935185185185186;
  gamma[9][4][9] = 0.00011574074074074075;
  gamma[9][5][4] = 0.00011574074074074075;
  gamma[9][5][5] = 0.001244212962962963;
  gamma[9][5][6] = 0.0007233796296296296;
  gamma[9][5][7] = 0.00011574074074074075;
  gamma[9][5][8] = 0.00011574074074074075;
  gamma[9][5][9] = 0.0007233796296296296;
  gamma[9][6][4] = 0.00011574074074074075;
  gamma[9][6][5] = 0.0007233796296296296;
  gamma[9][6][6] = 0.001244212962962963;
  gamma[9][6][7] = 0.00011574074074074075;
  gamma[9][6][8] = 0.00011574074074074075;
  gamma[9][6][9] = 0.0007233796296296296;
  gamma[9][7][4] = 0.000028935185185185186;
  gamma[9][7][5] = 0.00011574074074074075;
  gamma[9][7][6] = 0.00011574074074074075;
  gamma[9][7][7] = 0.000028935185185185186;
  gamma[9][7][8] = 0.000028935185185185186;
  gamma[9][7][9] = 0.00011574074074074075;
  gamma[9][8][4] = 0.000028935185185185186;
  gamma[9][8][5] = 0.00011574074074074075;
  gamma[9][8][6] = 0.00011574074074074075;
  gamma[9][8][7] = 0.000028935185185185186;
  gamma[9][8][8] = 0.000028935185185185186;
  gamma[9][8][9] = 0.00011574074074074075;
  gamma[9][9][4] = 0.00011574074074074075;
  gamma[9][9][5] = 0.0007233796296296296;
  gamma[9][9][6] = 0.0007233796296296296;
  gamma[9][9][7] = 0.00011574074074074075;
  gamma[9][9][8] = 0.00011574074074074075;
  gamma[9][9][9] = 0.001244212962962963;
  gamma[10][4][4] = 0.000028935185185185186;
  gamma[10][4][5] = 0.000028935185185185186;
  gamma[10][4][6] = 0.00011574074074074075;
  gamma[10][4][7] = 0.00011574074074074075;
  gamma[10][4][8] = 0.000028935185185185186;
  gamma[10][4][9] = 0.00011574074074074075;
  gamma[10][5][4] = 0.000028935185185185186;
  gamma[10][5][5] = 0.000028935185185185186;
  gamma[10][5][6] = 0.00011574074074074075;
  gamma[10][5][7] = 0.00011574074074074075;
  gamma[10][5][8] = 0.000028935185185185186;
  gamma[10][5][9] = 0.00011574074074074075;
  gamma[10][6][4] = 0.00011574074074074075;
  gamma[10][6][5] = 0.00011574074074074075;
  gamma[10][6][6] = 0.001244212962962963;
  gamma[10][6][7] = 0.0007233796296296296;
  gamma[10][6][8] = 0.00011574074074074075;
  gamma[10][6][9] = 0.0007233796296296296;
  gamma[10][7][4] = 0.00011574074074074075;
  gamma[10][7][5] = 0.00011574074074074075;
  gamma[10][7][6] = 0.0007233796296296296;
  gamma[10][7][7] = 0.001244212962962963;
  gamma[10][7][8] = 0.00011574074074074075;
  gamma[10][7][9] = 0.0007233796296296296;
  gamma[10][8][4] = 0.000028935185185185186;
  gamma[10][8][5] = 0.000028935185185185186;
  gamma[10][8][6] = 0.00011574074074074075;
  gamma[10][8][7] = 0.00011574074074074075;
  gamma[10][8][8] = 0.000028935185185185186;
  gamma[10][8][9] = 0.00011574074074074075;
  gamma[10][9][4] = 0.00011574074074074075;
  gamma[10][9][5] = 0.00011574074074074075;
  gamma[10][9][6] = 0.0007233796296296296;
  gamma[10][9][7] = 0.0007233796296296296;
  gamma[10][9][8] = 0.00011574074074074075;
  gamma[10][9][9] = 0.001244212962962963;
  gamma[11][4][4] = 0.001244212962962963;
  gamma[11][4][5] = 0.00011574074074074075;
  gamma[11][4][6] = 0.0007233796296296296;
  gamma[11][4][7] = 0.0007233796296296296;
  gamma[11][4][8] = 0.00011574074074074075;
  gamma[11][4][9] = 0.00011574074074074075;
  gamma[11][5][4] = 0.00011574074074074075;
  gamma[11][5][5] = 0.000028935185185185186;
  gamma[11][5][6] = 0.00011574074074074075;
  gamma[11][5][7] = 0.00011574074074074075;
  gamma[11][5][8] = 0.000028935185185185186;
  gamma[11][5][9] = 0.000028935185185185186;
  gamma[11][6][4] = 0.0007233796296296296;
  gamma[11][6][5] = 0.00011574074074074075;
  gamma[11][6][6] = 0.001244212962962963;
  gamma[11][6][7] = 0.0007233796296296296;
  gamma[11][6][8] = 0.00011574074074074075;
  gamma[11][6][9] = 0.00011574074074074075;
  gamma[11][7][4] = 0.0007233796296296296;
  gamma[11][7][5] = 0.00011574074074074075;
  gamma[11][7][6] = 0.0007233796296296296;
  gamma[11][7][7] = 0.001244212962962963;
  gamma[11][7][8] = 0.00011574074074074075;
  gamma[11][7][9] = 0.00011574074074074075;
  gamma[11][8][4] = 0.00011574074074074075;
  gamma[11][8][5] = 0.000028935185185185186;
  gamma[11][8][6] = 0.00011574074074074075;
  gamma[11][8][7] = 0.00011574074074074075;
  gamma[11][8][8] = 0.000028935185185185186;
  gamma[11][8][9] = 0.000028935185185185186;
  gamma[11][9][4] = 0.00011574074074074075;
  gamma[11][9][5] = 0.000028935185185185186;
  gamma[11][9][6] = 0.00011574074074074075;
  gamma[11][9][7] = 0.00011574074074074075;
  gamma[11][9][8] = 0.000028935185185185186;
  gamma[11][9][9] = 0.000028935185185185186;
}

HPC_NOINLINE HPC_HOST_DEVICE inline void get_O(hpc::array<hpc::vector3<double>, 10> const& x, S_t const& S, O_t& O) noexcept {
  for (int tet = 0; tet < 12; ++tet) {
    O[tet] = hpc::matrix3x3<double>::zero();
  }
  for (int tet = 0; tet < 12; ++tet) {
    for (int node = 0; node < 10; ++node) {
      for (int dim1 = 0; dim1 < 3; ++dim1) {
        for (int dim2 = 0; dim2 < 3; ++dim2) {
          O[tet](dim1, dim2) += x[node](dim1) * S[tet][node](dim2);
        }
      }
    }
  }
}

HPC_NOINLINE HPC_HOST_DEVICE inline void get_O_det(O_t const& O, hpc::array<double, 12>& det_O) noexcept {
  for (int tet = 0; tet < 12; ++tet) {
    det_O[tet] = determinant(O[tet]);
  }
}

static HPC_ALWAYS_INLINE HPC_HOST_DEVICE hpc::array<double, 4> get_barycentric(hpc::vector3<double> const x) noexcept {
  hpc::array<double, 4> xi;
  xi[0] = 1.0 - x(0) - x(1) - x(2);
  xi[1] = x(0);
  xi[2] = x(1);
  xi[3] = x(2);
  return xi;
}

HPC_NOINLINE HPC_HOST_DEVICE inline void get_ref_points(hpc::array<hpc::vector3<double>, 4>& pts) noexcept {
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
}

HPC_NOINLINE HPC_HOST_DEVICE inline void get_subtet_int(subtet_int_t& subtet_int) noexcept {
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
}

}
}