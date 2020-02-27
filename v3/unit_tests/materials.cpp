#include <iostream>
#include <gtest/gtest.h>
#include <j2/hardening.hpp>
#include "../otm_materials.hpp"


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

  auto sigma(hpc::matrix3x3<double>::zero());
  double Keff, Geff, W;

  lgr::neo_Hookean_point(F, K, G, sigma, Keff, Geff, W);

  auto dummy_stress(hpc::matrix3x3<double>::zero());
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
  hpc::matrix3x3<double> sigma_h(1.0/hpc::determinant(F)*P_h*hpc::transpose(F));

  auto error = hpc::norm(sigma - sigma_h)/hpc::norm(sigma_h);
  auto const eps = 10.0 * h*h;

  ASSERT_LE(error, eps);
}

TEST(materials, J2_point_consistency)
{
  std::cout.precision(12);
  double const K{(400.0/3.0)*1e9};
  double const G{80.0e9};
  double const Y0{350e6};
  double const n{4.0};
  double const eps0{1e-2};
  double const Svis0{Y0};
  double const m{2.0};
  double const eps_dot0{1e-1};
  lgr::j2::Properties const props{ .K = K, .G = G, .Y0 = Y0,
                                   .n = n, .eps0 = eps0, .Svis0 = Svis0,
                                   .m = m, .eps_dot0 = eps_dot0 };

  double dt = 1.0;

  auto F(hpc::matrix3x3<double>::identity());
  F(0,0) =  4.487668105802061e-01;
  F(0,1) =  3.518970848025007e-01;
  F(0,2) =  8.164609618407068e-01;
  F(1,0) = -8.254793765639534e-01;
  F(1,1) =  5.645538419297988e-01;
  F(1,2) =  1.692966029173534e-01;
  F(2,0) = -3.667724595991901e-01;
  F(2,1) = -7.456034473500678e-01;
  F(2,2) =  5.420598320853882e-01;

  auto Fp(hpc::matrix3x3<double>::identity());
  Fp(0,0) =  4.463159537096824e-01;
  Fp(0,1) =  3.561859252816812e-01;
  Fp(0,2) =  8.168361447751614e-01;
  Fp(1,0) = -8.215608636281700e-01;
  Fp(1,1) =  5.613285804297486e-01;
  Fp(1,2) =  1.740541995649333e-01;
  Fp(2,0) = -3.712031484597964e-01;
  Fp(2,1) = -7.457840032128831e-01;
  Fp(2,2) =  5.420064153795635e-01;

  double eqps = 2.145536499224904e-2;

  auto sigma(hpc::matrix3x3<double>::zero());
  double Keff, Geff, W;

  hpc::matrix3x3<double> Fp_new{Fp};
  double eqps_new{eqps};
  lgr::variational_J2_point(F, props, dt, sigma, Keff, Geff, W, Fp_new, eqps_new);

  auto P_h(hpc::matrix3x3<double>::zero());
  auto stress_dummy(hpc::matrix3x3<double>::zero());
  double Wp(0), Wm(0);
  double h = 1e-5;
  for (int i=0; i<3; ++i) {
    for (int j=0; j<3; ++j) {
      F(i,j) += h;
      hpc::matrix3x3<double> Fp_dummy{Fp};
      double eqps_dummy{eqps};
      lgr::variational_J2_point(F, props, dt, stress_dummy, Keff, Geff, Wp, Fp_dummy, eqps_dummy);

      F(i,j) -= 2.0*h;
      Fp_dummy = Fp;
      eqps_dummy = eqps;
      lgr::variational_J2_point(F, props, dt, stress_dummy, Keff, Geff, Wm, Fp_dummy, eqps_dummy);

      F(i,j) += h;

      P_h(i,j) = (Wp - Wm)/(2.0*h);
    }
  }
  hpc::matrix3x3<double> sigma_h(1.0/hpc::determinant(F)*P_h*hpc::transpose(F));

  auto error = hpc::norm(sigma - sigma_h)/hpc::norm(sigma_h);
  auto const tol = 5e3 * h*h;

  ASSERT_LE(error, tol);
}


TEST(materials, power_law_hardening)
{
  double const K{1.0e9};
  double const G{1.0e9};
  double const Y0{10e9};
  double const n{2.0};
  double const eps0{1e-3};
  double const Svis0{0};
  double const m{1.0};
  double const eps_dot0{1e-3};
  lgr::j2::Properties const props{ .K = K, .G = G, .Y0 = Y0,
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
  double const Y0{10e9};
  double const n{2.0};
  double const eps0{1e-3};
  double const Svis0{10e6};
  double const m{1.0};
  double const eps_dot0{1e-3};
  lgr::j2::Properties const props{ .K = K, .G = G, .Y0 = Y0,
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
