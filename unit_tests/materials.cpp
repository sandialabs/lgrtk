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

TEST(materials, J2_point_consistency)
{
  double const K{1.0e9};
  double const G{1.0e9};
  double const S0{10e9};
  double const n{2.0};
  double const eps0{1e-3};
  double const Svis0{0};
  double const m{1.0};
  double const eps_dot0{1e-3};
  lgr::j2::Properties const props{ .K = K, .G = G, .S0 = S0,
                                   .n = n, .eps0 = eps0, .Svis0 = Svis0,
                                   .m = m, .eps_dot0 = eps_dot0 };

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

  auto Fp(hpc::matrix3x3<double>::identity());
  Fp(0,0) *= 1.25;
  Fp(1,1) *= 2.0;
  Fp(2,2) *= 0.4;
  PrintMatrix(Fp);


  auto sigma(hpc::symmetric3x3<double>::zero());
  double Keff, Geff, W;

  lgr::variational_J2_point(F, props, sigma, Keff, Geff, W, Fp);

  auto dummy_stress(hpc::symmetric3x3<double>::zero());
  auto P_h(hpc::matrix3x3<double>::zero());
  double Wp(0), Wm(0);
  double h = 1e-5;
  for (int i=0; i<3; ++i) {
    for (int j=0; j<3; ++j) {
      F(i,j) += h;
      lgr::variational_J2_point(F, props, dummy_stress, Keff, Geff, Wp, Fp);

      F(i,j) -= 2*h;
      lgr::variational_J2_point(F, props, dummy_stress, Keff, Geff, Wm, Fp);

      F(i,j) += h;

      P_h(i,j) = (Wp - Wm)/(2.0*h);
    }
  }
  hpc::symmetric3x3<double> sigma_h(1.0/hpc::determinant(F)*P_h*hpc::transpose(F));

  auto error = hpc::norm(sigma - sigma_h)/hpc::norm(sigma_h);
  auto const tol = 10.0 * h*h;

  PrintMatrix(sigma);
  PrintMatrix(sigma_h);

  ASSERT_LE(error, tol);
}


TEST(materials, power_law_hardening)
{
  double const K{1.0e9};
  double const G{1.0e9};
  double const S0{10e9};
  double const n{2.0};
  double const eps0{1e-3};
  double const Svis0{0};
  double const m{1.0};
  double const eps_dot0{1e-3};
  lgr::j2::Properties const props{ .K = K, .G = G, .S0 = S0,
                                   .n = n, .eps0 = eps0, .Svis0 = Svis0,
                                   .m = m, .eps_dot0 = eps_dot0 };

  double eqps = 0.685;

  //double W = lgr::j2::HardeningPotential(props, eqps);
  double S = lgr::j2::FlowStrength(props, eqps);
  double H = lgr::j2::HardeningRate(props, eqps);

  double h = 1e-5;

  eqps += h;
  double Wp = lgr::j2::HardeningPotential(props, eqps);
  double Sp = lgr::j2::FlowStrength(props, eqps);

  eqps -= 2*h;
  double Wm = lgr::j2::HardeningPotential(props, eqps);
  double Sm = lgr::j2::FlowStrength(props, eqps);

  eqps += h;

  double S_h = 0.5*(Wp - Wm)/h;
  double H_h = 0.5*(Sp - Sm)/h;

  double error1 = std::abs(S_h - S)/S_h;
  double error2 = std::abs(H_h - H)/H_h;

  double error = std::max(error1, error2);
  double tol = 10.0*h*h  ;

  ASSERT_LE(error, tol);
}

TEST(materials, power_law_rate_sensitivity)
{
  double const K{1.0e9};
  double const G{1.0e9};
  double const S0{10e9};
  double const n{2.0};
  double const eps0{1e-3};
  double const Svis0{10e6};
  double const m{1.0};
  double const eps_dot0{1e-3};
  lgr::j2::Properties const props{ .K = K, .G = G, .S0 = S0,
                                   .n = n, .eps0 = eps0, .Svis0 = Svis0,
                                   .m = m, .eps_dot0 = eps_dot0 };

  double delta_eqps = 0.5;
  double const dt = 0.01;

  double Svis = lgr::j2::ViscoplasticStress(props, delta_eqps, dt);
  double Hvis = lgr::j2::ViscoplasticHardeningRate(props, delta_eqps, dt);

  double const h = 1e-5;

  delta_eqps += h;
  double Psi_star_p = lgr::j2::ViscoplasticDualKineticPotential(props, delta_eqps, dt);
  double Svis_p = lgr::j2::ViscoplasticStress(props, delta_eqps, dt);

  delta_eqps -= 2*h;
  double Psi_star_m = lgr::j2::ViscoplasticDualKineticPotential(props, delta_eqps, dt);
  double Svis_m = lgr::j2::ViscoplasticStress(props, delta_eqps, dt);

  delta_eqps += h;

  double Svis_h = 0.5*(Psi_star_p - Psi_star_m)/h;
  double Hvis_h = 0.5*(Svis_p - Svis_m)/h;

  double error1 = std::abs(Svis_h - Svis)/Svis_h;
  double error2 = std::abs(Hvis_h - Hvis)/Hvis_h;

  double error = std::max(error1, error2);
  double tol = 10.0*h*h  ;

  ASSERT_LE(error, tol);
}
