#ifndef LGR_COMPTET_FUNCTIONS_HPP
#define LGR_COMPTET_FUNCTIONS_HPP

OMEGA_H_INLINE
double CompTet::det_44(Matrix<4, 4> a) {
  double det =
    a(0, 3) * a(1, 2) * a(2, 1) * a(3, 0) - a(0, 2) * a(1, 3) * a(2, 1) * a(3, 0) -
    a(0, 3) * a(1, 1) * a(2, 2) * a(3, 0) + a(0, 1) * a(1, 3) * a(2, 2) * a(3, 0) +
    a(0, 2) * a(1, 1) * a(2, 3) * a(3, 0) - a(0, 1) * a(1, 2) * a(2, 3) * a(3, 0) -
    a(0, 3) * a(1, 2) * a(2, 0) * a(3, 1) + a(0, 2) * a(1, 3) * a(2, 0) * a(3, 1) +
    a(0, 3) * a(1, 0) * a(2, 2) * a(3, 1) - a(0, 0) * a(1, 3) * a(2, 2) * a(3, 1) -
    a(0, 2) * a(1, 0) * a(2, 3) * a(3, 1) + a(0, 0) * a(1, 2) * a(2, 3) * a(3, 1) +
    a(0, 3) * a(1, 1) * a(2, 0) * a(3, 2) - a(0, 1) * a(1, 3) * a(2, 0) * a(3, 2) -
    a(0, 3) * a(1, 0) * a(2, 1) * a(3, 2) + a(0, 0) * a(1, 3) * a(2, 1) * a(3, 2) +
    a(0, 1) * a(1, 0) * a(2, 3) * a(3, 2) - a(0, 0) * a(1, 1) * a(2, 3) * a(3, 2) -
    a(0, 2) * a(1, 1) * a(2, 0) * a(3, 3) + a(0, 1) * a(1, 2) * a(2, 0) * a(3, 3) +
    a(0, 2) * a(1, 0) * a(2, 1) * a(3, 3) - a(0, 0) * a(1, 2) * a(2, 1) * a(3, 3) -
    a(0, 1) * a(1, 0) * a(2, 2) * a(3, 3) + a(0, 0) * a(1, 1) * a(2, 2) * a(3, 3);
  return det;
}

OMEGA_H_INLINE
Matrix<4, 4> CompTet::invert_44(Matrix<4, 4> a) {
  Matrix<4, 4> ai;
  double xj = det_44(a);
  ai(0, 0) = (1 / xj) * (-a(1, 3) * a(2, 2) * a(3, 1) + a(1, 2) * a(2, 3) * a(3, 1) + a(1, 3) * a(2, 1) * a(3, 2) -
                         a(1, 1) * a(2, 3) * a(3, 2) - a(1, 2) * a(2, 1) * a(3, 3) + a(1, 1) * a(2, 2) * a(3, 3));
  ai(0, 1) = (1 / xj) * (a(0, 3) * a(2, 2) * a(3, 1) - a(0, 2) * a(2, 3) * a(3, 1) - a(0, 3) * a(2, 1) * a(3, 2) +
                         a(0, 1) * a(2, 3) * a(3, 2) + a(0, 2) * a(2, 1) * a(3, 3) - a(0, 1) * a(2, 2) * a(3, 3));
  ai(0, 2) = (1 / xj) * (-a(0, 3) * a(1, 2) * a(3, 1) + a(0, 2) * a(1, 3) * a(3, 1) + a(0, 3) * a(1, 1) * a(3, 2) -
                         a(0, 1) * a(1, 3) * a(3, 2) - a(0, 2) * a(1, 1) * a(3, 3) + a(0, 1) * a(1, 2) * a(3, 3));
  ai(0, 3) = (1 / xj) * (a(0, 3) * a(1, 2) * a(2, 1) - a(0, 2) * a(1, 3) * a(2, 1) - a(0, 3) * a(1, 1) * a(2, 2) +
                         a(0, 1) * a(1, 3) * a(2, 2) + a(0, 2) * a(1, 1) * a(2, 3) - a(0, 1) * a(1, 2) * a(2, 3));
  ai(1, 0) = (1 / xj) * (a(1, 3) * a(2, 2) * a(3, 0) - a(1, 2) * a(2, 3) * a(3, 0) - a(1, 3) * a(2, 0) * a(3, 2) +
                         a(1, 0) * a(2, 3) * a(3, 2) + a(1, 2) * a(2, 0) * a(3, 3) - a(1, 0) * a(2, 2) * a(3, 3));
  ai(1, 1) = (1 / xj) * (-a(0, 3) * a(2, 2) * a(3, 0) + a(0, 2) * a(2, 3) * a(3, 0) + a(0, 3) * a(2, 0) * a(3, 2) -
                         a(0, 0) * a(2, 3) * a(3, 2) - a(0, 2) * a(2, 0) * a(3, 3) + a(0, 0) * a(2, 2) * a(3, 3));
  ai(1, 2) = (1 / xj) * (a(0, 3) * a(1, 2) * a(3, 0) - a(0, 2) * a(1, 3) * a(3, 0) - a(0, 3) * a(1, 0) * a(3, 2) +
                         a(0, 0) * a(1, 3) * a(3, 2) + a(0, 2) * a(1, 0) * a(3, 3) - a(0, 0) * a(1, 2) * a(3, 3));
  ai(1, 3) = (1 / xj) * (-a(0, 3) * a(1, 2) * a(2, 0) + a(0, 2) * a(1, 3) * a(2, 0) + a(0, 3) * a(1, 0) * a(2, 2) -
                         a(0, 0) * a(1, 3) * a(2, 2) - a(0, 2) * a(1, 0) * a(2, 3) + a(0, 0) * a(1, 2) * a(2, 3));
  ai(2, 0) = (1 / xj) * (-a(1, 3) * a(2, 1) * a(3, 0) + a(1, 1) * a(2, 3) * a(3, 0) + a(1, 3) * a(2, 0) * a(3, 1) -
                         a(1, 0) * a(2, 3) * a(3, 1) - a(1, 1) * a(2, 0) * a(3, 3) + a(1, 0) * a(2, 1) * a(3, 3));
  ai(2, 1) = (1 / xj) * (a(0, 3) * a(2, 1) * a(3, 0) - a(0, 1) * a(2, 3) * a(3, 0) - a(0, 3) * a(2, 0) * a(3, 1) +
                         a(0, 0) * a(2, 3) * a(3, 1) + a(0, 1) * a(2, 0) * a(3, 3) - a(0, 0) * a(2, 1) * a(3, 3));
  ai(2, 2) = (1 / xj) * (-a(0, 3) * a(1, 1) * a(3, 0) + a(0, 1) * a(1, 3) * a(3, 0) + a(0, 3) * a(1, 0) * a(3, 1) -
                         a(0, 0) * a(1, 3) * a(3, 1) - a(0, 1) * a(1, 0) * a(3, 3) + a(0, 0) * a(1, 1) * a(3, 3));
  ai(2, 3) = (1 / xj) * (a(0, 3) * a(1, 1) * a(2, 0) - a(0, 1) * a(1, 3) * a(2, 0) - a(0, 3) * a(1, 0) * a(2, 1) +
                         a(0, 0) * a(1, 3) * a(2, 1) + a(0, 1) * a(1, 0) * a(2, 3) - a(0, 0) * a(1, 1) * a(2, 3));
  ai(3, 0) = (1 / xj) * (a(1, 2) * a(2, 1) * a(3, 0) - a(1, 1) * a(2, 2) * a(3, 0) - a(1, 2) * a(2, 0) * a(3, 1) +
                         a(1, 0) * a(2, 2) * a(3, 1) + a(1, 1) * a(2, 0) * a(3, 2) - a(1, 0) * a(2, 1) * a(3, 2));
  ai(3, 1) = (1 / xj) * (-a(0, 2) * a(2, 1) * a(3, 0) + a(0, 1) * a(2, 2) * a(3, 0) + a(0, 2) * a(2, 0) * a(3, 1) -
                         a(0, 0) * a(2, 2) * a(3, 1) - a(0, 1) * a(2, 0) * a(3, 2) + a(0, 0) * a(2, 1) * a(3, 2));
  ai(3, 2) = (1 / xj) * (a(0, 2) * a(1, 1) * a(3, 0) - a(0, 1) * a(1, 2) * a(3, 0) - a(0, 2) * a(1, 0) * a(3, 1) +
                         a(0, 0) * a(1, 2) * a(3, 1) + a(0, 1) * a(1, 0) * a(3, 2) - a(0, 0) * a(1, 1) * a(3, 2));
  ai(3, 3) = (1 / xj) * (-a(0, 2) * a(1, 1) * a(2, 0) + a(0, 1) * a(1, 2) * a(2, 0) + a(0, 2) * a(1, 0) * a(2, 1) -
                         a(0, 0) * a(1, 2) * a(2, 1) - a(0, 1) * a(1, 0) * a(2, 2) + a(0, 0) * a(1, 1) * a(2, 2));
  return ai;
}

template <int NI, int NJ, int NK>
OMEGA_H_INLINE
void CompTet::zero(Omega_h::Few<Omega_h::Matrix<NJ, NK>, NI> in) {
  for (int i = 0; i < NI; ++i) {
    for (int j = 0; j < NJ; ++j) {
      for (int k = 0; k < NK; ++k) {
        in[i](j, k) = 0.0;
      }
    }
  }
}

OMEGA_H_INLINE
CompTet::SOptType CompTet::compute_S_opt() {
  SOptType S_opt;
  S_opt[0](0, 0) = -2;
  S_opt[0](0, 1) = -2;
  S_opt[0](0, 2) = -2;
  S_opt[0](4, 0) = 2;
  S_opt[0](6, 1) = 2;
  S_opt[0](7, 2) = 2;
  S_opt[1](1, 0) = 2;
  S_opt[1](4, 0) = -2;
  S_opt[1](4, 1) = -2;
  S_opt[1](4, 2) = -2;
  S_opt[1](5, 1) = 2;
  S_opt[1](8, 2) = 2;
  S_opt[2](2, 1) = 2;
  S_opt[2](5, 0) = 2;
  S_opt[2](6, 0) = -2;
  S_opt[2](6, 1) = -2;
  S_opt[2](6, 2) = -2;
  S_opt[2](9, 2) = 2;
  S_opt[3](3, 2) = 2;
  S_opt[3](7, 0) = -2;
  S_opt[3](7, 1) = -2;
  S_opt[3](7, 2) = -2;
  S_opt[3](8, 0) = 2;
  S_opt[3](9, 1) = 2;
  S_opt[4](4, 0) = -0.6666666666666666;
  S_opt[4](4, 1) = -2;
  S_opt[4](4, 2) = -2;
  S_opt[4](5, 0) = 1.3333333333333333;
  S_opt[4](5, 1) = 2;
  S_opt[4](6, 0) = -0.6666666666666666;
  S_opt[4](7, 0) = -0.6666666666666666;
  S_opt[4](8, 0) = 1.3333333333333333;
  S_opt[4](8, 2) = 2;
  S_opt[4](9, 0) = -0.6666666666666666;
  S_opt[5](4, 0) = -0.6666666666666666;
  S_opt[5](4, 1) = -0.6666666666666666;
  S_opt[5](4, 2) = -0.6666666666666666;
  S_opt[5](5, 0) = 1.3333333333333333;
  S_opt[5](5, 1) = 1.3333333333333333;
  S_opt[5](5, 2) = -0.6666666666666666;
  S_opt[5](6, 0) = -0.6666666666666666;
  S_opt[5](6, 1) = -0.6666666666666666;
  S_opt[5](6, 2) = -0.6666666666666666;
  S_opt[5](7, 0) = -0.6666666666666666;
  S_opt[5](7, 1) = -0.6666666666666666;
  S_opt[5](7, 2) = -0.6666666666666666;
  S_opt[5](8, 0) = 1.3333333333333333;
  S_opt[5](8, 1) = -0.6666666666666666;
  S_opt[5](8, 2) = 1.3333333333333333;
  S_opt[5](9, 0) = -0.6666666666666666;
  S_opt[5](9, 1) = 1.3333333333333333;
  S_opt[5](9, 2) = 1.3333333333333333;
  S_opt[6](4, 2) = -0.6666666666666666;
  S_opt[6](5, 2) = -0.6666666666666666;
  S_opt[6](6, 2) = -0.6666666666666666;
  S_opt[6](7, 0) = -2;
  S_opt[6](7, 1) = -2;
  S_opt[6](7, 2) = -0.6666666666666666;
  S_opt[6](8, 0) = 2;
  S_opt[6](8, 2) = 1.3333333333333333;
  S_opt[6](9, 1) = 2;
  S_opt[6](9, 2) = 1.3333333333333333;
  S_opt[7](4, 1) = -1.3333333333333333;
  S_opt[7](4, 2) = -2;
  S_opt[7](5, 1) = 0.6666666666666666;
  S_opt[7](6, 1) = 0.6666666666666666;
  S_opt[7](7, 0) = -2;
  S_opt[7](7, 1) = -1.3333333333333333;
  S_opt[7](8, 0) = 2;
  S_opt[7](8, 1) = 0.6666666666666666;
  S_opt[7](8, 2) = 2;
  S_opt[7](9, 1) = 0.6666666666666666;
  S_opt[8](4, 1) = -2;
  S_opt[8](4, 2) = -1.3333333333333333;
  S_opt[8](5, 0) = 2;
  S_opt[8](5, 1) = 2;
  S_opt[8](5, 2) = 0.6666666666666666;
  S_opt[8](6, 0) = -2;
  S_opt[8](6, 2) = -1.3333333333333333;
  S_opt[8](7, 2) = 0.6666666666666666;
  S_opt[8](8, 2) = 0.6666666666666666;
  S_opt[8](9, 2) = 0.6666666666666666;
  S_opt[9](4, 1) = -0.6666666666666666;
  S_opt[9](5, 0) = 2;
  S_opt[9](5, 1) = 1.3333333333333333;
  S_opt[9](6, 0) = -2;
  S_opt[9](6, 1) = -0.6666666666666666;
  S_opt[9](6, 2) = -2;
  S_opt[9](7, 1) = -0.6666666666666666;
  S_opt[9](8, 1) = -0.6666666666666666;
  S_opt[9](9, 1) = 1.3333333333333333;
  S_opt[9](9, 2) = 2;
  S_opt[10](4, 0) = 0.6666666666666666;
  S_opt[10](5, 0) = 0.6666666666666666;
  S_opt[10](6, 0) = -1.3333333333333333;
  S_opt[10](6, 2) = -2;
  S_opt[10](7, 0) = -1.3333333333333333;
  S_opt[10](7, 1) = -2;
  S_opt[10](8, 0) = 0.6666666666666666;
  S_opt[10](9, 0) = 0.6666666666666666;
  S_opt[10](9, 1) = 2;
  S_opt[10](9, 2) = 2;
  S_opt[11](4, 0) = 0.6666666666666666;
  S_opt[11](4, 1) = -1.3333333333333333;
  S_opt[11](4, 2) = -1.3333333333333333;
  S_opt[11](5, 0) = 0.6666666666666666;
  S_opt[11](5, 1) = 0.6666666666666666;
  S_opt[11](5, 2) = 0.6666666666666666;
  S_opt[11](6, 0) = -1.3333333333333333;
  S_opt[11](6, 1) = 0.6666666666666666;
  S_opt[11](6, 2) = -1.3333333333333333;
  S_opt[11](7, 0) = -1.3333333333333333;
  S_opt[11](7, 1) = -1.3333333333333333;
  S_opt[11](7, 2) = 0.6666666666666666;
  S_opt[11](8, 0) = 0.6666666666666666;
  S_opt[11](8, 1) = 0.6666666666666666;
  S_opt[11](8, 2) = 0.6666666666666666;
  S_opt[11](9, 0) = 0.6666666666666666;
  S_opt[11](9, 1) = 0.6666666666666666;
  S_opt[11](9, 2) = 0.6666666666666666;
  return S_opt;
}

OMEGA_H_INLINE
CompTet::STIPMType CompTet::compute_sub_tet_int_proj_M() {
  STIPMType sub_tet_int_proj_M;
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

OMEGA_H_INLINE
Matrix<CompTet::nsub_tets, 4> CompTet::compute_sub_tet_int() {
  Matrix<nsub_tets, 4> sub_tet_int;
  sub_tet_int(0, 0) = 0.013020833333333334;
  sub_tet_int(0, 1) = 0.0026041666666666665;
  sub_tet_int(0, 2) = 0.0026041666666666665;
  sub_tet_int(0, 3) = 0.0026041666666666665;
  sub_tet_int(1, 0) = 0.0026041666666666665;
  sub_tet_int(1, 1) = 0.013020833333333334;
  sub_tet_int(1, 2) = 0.0026041666666666665;
  sub_tet_int(1, 3) = 0.0026041666666666665;
  sub_tet_int(2, 0) = 0.0026041666666666665;
  sub_tet_int(2, 1) = 0.0026041666666666665;
  sub_tet_int(2, 2) = 0.013020833333333334;
  sub_tet_int(2, 3) = 0.0026041666666666665;
  sub_tet_int(3, 0) = 0.0026041666666666665;
  sub_tet_int(3, 1) = 0.0026041666666666665;
  sub_tet_int(3, 2) = 0.0026041666666666665;
  sub_tet_int(3, 3) = 0.013020833333333334;
  sub_tet_int(4, 0) = 0.001953125;
  sub_tet_int(4, 1) = 0.004557291666666667;
  sub_tet_int(4, 2) = 0.001953125;
  sub_tet_int(4, 3) = 0.001953125;
  sub_tet_int(5, 0) = 0.0006510416666666666;
  sub_tet_int(5, 1) = 0.0032552083333333335;
  sub_tet_int(5, 2) = 0.0032552083333333335;
  sub_tet_int(5, 3) = 0.0032552083333333335;
  sub_tet_int(6, 0) = 0.001953125;
  sub_tet_int(6, 1) = 0.001953125;
  sub_tet_int(6, 2) = 0.001953125;
  sub_tet_int(6, 3) = 0.004557291666666667;
  sub_tet_int(7, 0) = 0.0032552083333333335;
  sub_tet_int(7, 1) = 0.0032552083333333335;
  sub_tet_int(7, 2) = 0.0006510416666666666;
  sub_tet_int(7, 3) = 0.0032552083333333335;
  sub_tet_int(8, 0) = 0.0032552083333333335;
  sub_tet_int(8, 1) = 0.0032552083333333335;
  sub_tet_int(8, 2) = 0.0032552083333333335;
  sub_tet_int(8, 3) = 0.0006510416666666666;
  sub_tet_int(9, 0) = 0.001953125;
  sub_tet_int(9, 1) = 0.001953125;
  sub_tet_int(9, 2) = 0.004557291666666667;
  sub_tet_int(9, 3) = 0.001953125;
  sub_tet_int(10, 0) = 0.0032552083333333335;
  sub_tet_int(10, 1) = 0.0006510416666666666;
  sub_tet_int(10, 2) = 0.0032552083333333335;
  sub_tet_int(10, 3) = 0.0032552083333333335;
  sub_tet_int(11, 0) = 0.004557291666666667;
  sub_tet_int(11, 1) = 0.001953125;
  sub_tet_int(11, 2) = 0.001953125;
  sub_tet_int(11, 3) = 0.001953125;
  return sub_tet_int;
}

OMEGA_H_INLINE
Matrix<4, 4> CompTet::compute_parent_M_inv() {
  Matrix<4, 4> M_inv;
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      if (i == j) {
        M_inv(i, i) = 96.0;
      } else {
        M_inv(i, j) = -24.0;
      }
    }
  }
  return M_inv;
}

OMEGA_H_INLINE
CompTet::OType CompTet::compute_O(Matrix<dim, nodes> x, SOptType S_opt) {
  OType O;
  zero(O);
  for (int tet = 0; tet < nsub_tets; ++tet) {
    for (int node = 0; node < nodes; ++node) {
      for (int dim1 = 0; dim1 < dim; ++dim1) {
        for (int dim2 = 0; dim2 < dim; ++dim2) {
          O[tet](dim1, dim2) += x[node][dim1] * S_opt[tet](node, dim2);
        }
      }
    }
  }
  return O;
}

OMEGA_H_INLINE
CompTet::OType CompTet::compute_O_inv(OType O) {
  OType O_inv;
  zero(O_inv);
  for (int tet = 0; tet < nsub_tets; ++tet) {
    O_inv[tet] = Omega_h::invert(O[tet]);
  }
  return O_inv;
}

OMEGA_H_INLINE
Vector<CompTet::nsub_tets> CompTet::compute_O_det(OType O) {
  Vector<nsub_tets> det_O;
  for (int tet = 0; tet < nsub_tets; ++tet) {
    det_O[tet] = Omega_h::determinant(O[tet]);
  }
  return det_O;
}

OMEGA_H_INLINE
Matrix<4, 4> CompTet::compute_M_inv(Vector<nsub_tets> O_det) {
  auto M = Omega_h::zero_matrix<4, 4>();
  auto sub_tet_int_proj_M = compute_sub_tet_int_proj_M();
  for (int tet = 0; tet < nsub_tets; ++tet) {
    for (int i = 0; i < nbarycentric_coords; ++i) {
      for (int j = 0; j < nbarycentric_coords; ++j) {
        M(i, j) += O_det[tet] * sub_tet_int_proj_M[tet](i, j);
      }
    }
  }
  return invert_44(M);
}

OMEGA_H_INLINE
CompTet::SOLType CompTet::compute_SOL(
    Vector<nsub_tets> O_det,
    OType O_inv,
    Matrix<nsub_tets, 4> sub_tet_int,
    SOptType S_opt) {
  SOLType SOL;
  for (int tet = 0; tet < nsub_tets; ++tet) {
    for (int node = 0; node < nodes; ++node) {
      for (int dim1 = 0; dim1 < dim; ++dim1) {
        for (int dim2 = 0; dim2 < dim; ++dim2) {
          for (int pt = 0; pt < nbarycentric_coords; ++pt) {
            SOL[pt](node, dim1) +=
              O_det[tet] * S_opt[tet](node, dim2) * O_inv[tet](dim2, dim1) * sub_tet_int(tet, pt);
          }
        }
      }
    }
  }
  return SOL;
}

OMEGA_H_INLINE
Vector<4> CompTet::compute_DOL(Vector<nsub_tets> O_det,
    Matrix<nsub_tets, 4> sub_tet_int) {
  auto DOL = Omega_h::zero_vector<4>();
  for (int tet = 0; tet < nsub_tets; ++tet) {
    for (int pt = 0; pt < nbarycentric_coords; ++pt) {
      DOL[pt] += O_det[tet] * sub_tet_int(tet, pt);
    }
  }
  return DOL;
}

OMEGA_H_INLINE
Shape<CompTet> CompTet::shape(Matrix<dim, nodes> node_coords) {
  using Omega_h::Few;
  double lambda[nbarycentric_coords];
  auto S_opt = compute_S_opt();
  auto parent_M_inv = compute_parent_M_inv();
  auto sub_tet_int = compute_sub_tet_int();
  auto O = compute_O(node_coords, S_opt);
  auto O_inv = compute_O_inv(O);
  auto O_det = compute_O_det(O);
  auto M_inv = compute_M_inv(O_det);
  auto SOL = compute_SOL(O_det, O_inv, sub_tet_int, S_opt);
  auto DOL = compute_DOL(O_det, sub_tet_int);
  Shape<CompTet> out;
  // need referecnce integration points to compute the gradient and |J| * w
  // gotta dig through intrepid to find this
  (void)lambda;
  (void)O_det;
  (void)O_inv;
  (void)parent_M_inv;
  (void)sub_tet_int;
  (void)M_inv;
  (void)SOL;
  (void)DOL;
  out.lengths.time_step_length = 0.0; // this needs to change
  out.lengths.viscosity_length = 0.0; // this needs to change
  return out;
}

OMEGA_H_INLINE
constexpr double CompTet::lumping_factor(int /*node */) {
  // we need to consider that density can vary over the element and
  // the fact that the lumping factor can depend on nodal coordinates.
  // returning 0 for now will at least break everything so that it's
  // clear that this is incorrect.
  return 0.0;
}

OMEGA_H_INLINE Matrix<CompTet::nodes, CompTet::points> CompTet::basis_values() {
  // need to fill this in
  Matrix<nodes, points> out;
  return out;
}

#endif
