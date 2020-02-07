#include <iostream>
#include <gtest/gtest.h>
#include "materials.hpp"

void PrintMatrix(hpc::matrix3x3<double> const &A)
{
  for (int i=0; i<3; ++i) {
    for (int j=0; j<3; ++j) {
      std::cout << A(i,j) << " ";
    }
    std::cout << std::endl;
  }
}

void PrintMatrix(hpc::symmetric3x3<double> const &A)
{
  for (int i=0; i<3; ++i) {
    for (int j=0; j<3; ++j) {
      std::cout << A(i,j) << " ";
    }
    std::cout << std::endl;
  }
}


TEST(materials, neohookean_point_consistency)
{
  double const G {1.0};
  double const K {1.0};
  auto F(hpc::matrix3x3<double>::identity());
  F(0,0) = 1.2;
  F(0,1) = 0.05;
  F(0,2) = 0.733;
  F(1,0) = 0.66;
  F(1,1) = 1.18;
  F(1,2) = 0.23;
  F(2,0) = 0.0;
  F(2,1) = 0.41;
  F(2,2) = 0.91;
  //std::cout.precision(12);
  //std::cout << "F = \n";
  //PrintMatrix(F);

  auto sigma(hpc::symmetric3x3<double>::zero());
  double Keff, Geff, W;

  lgr::neo_Hookean_point(F, K, G, sigma, Keff, Geff, W);

  auto dummy_stress(hpc::symmetric3x3<double>::zero());
  auto P_h(hpc::matrix3x3<double>::zero());
  double Wp(0), Wm(0);
  double h = 1e-5;
  for (int i=0; i<3; ++i) {
    for (int j=0; j<3; ++j) {
      F(i,j) += h;
      lgr::neo_Hookean_point(F, K, G, dummy_stress, Keff, Geff, Wp);

      F(i,j) -= 2*h;
      lgr::neo_Hookean_point(F, K, G, dummy_stress, Keff, Geff, Wm);

      F(i,j) += h;

      P_h(i,j) = (Wp - Wm)/(2.0*h);
    }
  }
  hpc::symmetric3x3<double> sigma_h(1.0/hpc::determinant(F)*P_h*hpc::transpose(F));

  auto error = hpc::norm(sigma - sigma_h)/hpc::norm(sigma_h);
  auto const eps = 10.0 * h*h;

  ASSERT_LE(error, eps);
}
