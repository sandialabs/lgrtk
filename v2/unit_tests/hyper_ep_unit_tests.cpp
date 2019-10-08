#ifdef ACTUALLY_BUILD_THIS_TEST
#include <cmath>

#include <lgr_hyper_ep.hpp>
#include <lgr_math.hpp>
#include "lgr_gtest.hpp"
#if 0
#include <Teuchos_ParameterList.hpp>
#endif

namespace {

using scalar_type = double;
namespace Details = lgr::hyper_ep;
using lgr::Tensor;
using lgr::identity_tensor;
using lgr::zero_matrix;

#ifdef OMEGA_H_THROW
using invalid_argument = Omega_h::exception;
#else
using invalid_argument = std::invalid_argument;
#endif

namespace hyper_ep_utils {

static scalar_type copper_density() { return 8930.0; }

static Details::Properties copper_johnson_cook_props()
{
  Details::Properties props;
  props.elastic = Details::Elastic::NEO_HOOKEAN;
  // Properties
  props.E = 200.0e9;
  props.Nu = 0.333;
  // Johnson cook hardening
  props.hardening = Details::Hardening::JOHNSON_COOK;
  props.A = 8.970000E+08;
  props.B = 2.918700E+09;
  props.n = 3.100000E-01;
  // Temperature dependence
  props.C1 = 1.189813E-01;
  props.C2 = std::numeric_limits<scalar_type>::max();
  props.C3 = 1.090000E+00;
  // Rate dependence
  props.rate_dep = Details::RateDependence::JOHNSON_COOK;
  props.C4 = 2.500000E-02  ;
  props.ep_dot_0 = 1.0;

  // Damage
  props.damage = Details::Damage::JOHNSON_COOK;
  props.D0 = 0.0;
  props.DC = 0.8;
  props.D1 = 0.54;
  props.D2 = 4.89;
  props.D3 = -3.03;
  props.D4 = 0.0;
  props.D5 = 0.0;
  props.set_stress_to_zero = false;
  props.allow_no_shear = false;
  props.allow_no_tension = true;
  props.eps_f_min = 0.6;

  return props;
}

static Details::Properties copper_zerilli_armstrong_props()
{
  Details::Properties props;
  // Properties
  props.elastic = Details::Elastic::NEO_HOOKEAN;
  props.E = 200.0e9;
  props.Nu = 0.333;
  // Constant yield strength
  props.hardening = Details::Hardening::ZERILLI_ARMSTRONG;
  props.A = 6.500000E+08;
  // Power law hardening
  props.B = 0.000000E+00;
  props.n = 1.000000E+00;
  //
  props.C1 = 0.000000E+00;
  props.C2 = 8.900000E+09;
  props.C3 = 3.249400E+01;
  // Rate dependence
  props.rate_dep = Details::RateDependence::ZERILLI_ARMSTRONG;
  props.C4 = 1.334575E+00;
  return props;
}

// Evaluate the material model through various prescribed motions. The
// correctness of output *is not checked*. We are just making sure that the
// model runs with the parameters.
static void eval_prescribed_motions(
    const scalar_type eps,
    Details::Properties const props,
    const scalar_type& rho)
{

  auto const E = props.E;
  auto const Nu = props.Nu;
  auto const bulk_modulus = E / 3.0 / (1.0 - 2.0 * Nu);
  auto const shear_modulus = E/2./(1.+Nu);
  auto const wave_speed_expected = std::sqrt((bulk_modulus+(4./3.)*shear_modulus)/rho);

  scalar_type dtime = 1.;
  scalar_type temp = 298.;

  // Initialize in/out variables to be updated by material update
  Tensor<3> T;
  Tensor<3> F = identity_tensor<3>();
  Tensor<3> Fp = identity_tensor<3>();

  scalar_type wave_speed = 0.;
  scalar_type ep = 0.;
  scalar_type epdot = 0.;
  scalar_type dp = 0.;
  scalar_type localized = 0.;

  Details::ErrorCode err;

  // Uniaxial strain, tension
  F(0,0) = 1. + eps; F(0,1) = 0.;       F(0,2) = 0.;
  F(1,0) = 0.;       F(1,1) = 1.;       F(1,2) = 0.;
  F(2,0) = 0.;       F(2,1) = 0.;       F(2,2) = 1.;
  err = Details::update(props, rho, F, dtime, temp, T, wave_speed,
                        Fp, ep, epdot, dp, localized);
  EXPECT_TRUE(err == Details::ErrorCode::SUCCESS)
    << "UNIAXIAL STRAIN, TENSION EVAL FAILED WITH ERROR "
    << "'" << Details::get_error_code_string(err) << "'";
  EXPECT_TRUE(Omega_h::are_close(wave_speed, wave_speed_expected))
    << "EXPECTED WAVE SPEED: " << wave_speed_expected << ", "
    << "CALCULATED WAVE SPEED: " << wave_speed;

  // Uniaxial strain, compression
  F(0,0) = 1. - eps; F(0,1) = 0.;       F(0,2) = 0.;
  F(1,0) = 0.;       F(1,1) = 1.;       F(1,2) = 0.;
  F(2,0) = 0.;       F(2,1) = 0.;       F(2,2) = 1.;
  err = Details::update(props, rho, F, dtime, temp, T, wave_speed,
                        Fp, ep, epdot, dp, localized);
  EXPECT_TRUE(err == Details::ErrorCode::SUCCESS)
    << "UNIAXIAL STRAIN, COMPRESSION EVAL FAILED WITH ERROR "
    << "'" << Details::get_error_code_string(err) << "'";

  // Simple shear, 2D
  F(0,0) = 1.;       F(0,1) = eps;      F(0,2) = 0.;
  F(1,0) = 0.;       F(1,1) = 1.;       F(1,2) = 0.;
  F(2,0) = 0.;       F(2,1) = 0.;       F(2,2) = 1.;
  err = Details::update(props, rho, F, dtime, temp, T, wave_speed,
                        Fp, ep, epdot, dp, localized);
  EXPECT_TRUE(err == Details::ErrorCode::SUCCESS)
    << "SIMPLE SHEAR, 2D EVAL FAILED WITH ERROR "
    << "'" << Details::get_error_code_string(err) << "'";

  // Hydrostatic compression
  F(0,0) = 1. - eps; F(0,1) = 0.;       F(0,2) = 0.;
  F(1,0) = 0.;       F(1,1) = 1. - eps; F(1,2) = 0.;
  F(2,0) = 0.;       F(2,1) = 0.;       F(2,2) = 1. - eps;
  err = Details::update(props, rho, F, dtime, temp, T, wave_speed,
                        Fp, ep, epdot, dp, localized);
  EXPECT_TRUE(err == Details::ErrorCode::SUCCESS)
    << "HYDROSTATIC COMPRESSION EVAL FAILED WITH ERROR "
    << "'" << Details::get_error_code_string(err) << "'";

  // Hydrostatic tension
  F(0,0) = 1. + eps; F(0,1) = 0.;       F(0,2) = 0.;
  F(1,0) = 0.;       F(1,1) = 1. + eps; F(1,2) = 0.;
  F(2,0) = 0.;       F(2,1) = 0.;       F(2,2) = 1. + eps;
  err = Details::update(props, rho, F, dtime, temp, T, wave_speed,
                        Fp, ep, epdot, dp, localized);
  EXPECT_TRUE(err == Details::ErrorCode::SUCCESS)
    << "HYDROSTATIC TENSION EVAL FAILED WITH ERROR "
    << "'" << Details::get_error_code_string(err) << "'";

  // Simple shear, 3D
  F(0,0) = 1.;       F(0,1) = eps;      F(0,2) = 0.;
  F(1,0) = eps;      F(1,1) = 1.;       F(1,2) = eps;
  F(2,0) = eps;      F(2,1) = 0.;       F(2,2) = 1.;
  err = Details::update(props, rho, F, dtime, temp, T, wave_speed,
                        Fp, ep, epdot, dp, localized);
  EXPECT_TRUE(err == Details::ErrorCode::SUCCESS)
    << "SIMPLE SHEAR, 3D EVAL FAILED WITH ERROR "
    << "'" << Details::get_error_code_string(err) << "'";

  // Biaxial strain, tension
  F(0,0) = 1. + eps; F(0,1) = 0.;       F(0,2) = 0.;
  F(1,0) = 0.;       F(1,1) = 1. + eps; F(1,2) = 0.;
  F(2,0) = 0.;       F(2,1) = 0.;       F(2,2) = 1.;
  err = Details::update(props, rho, F, dtime, temp, T, wave_speed,
                        Fp, ep, epdot, dp, localized);
  EXPECT_TRUE(err == Details::ErrorCode::SUCCESS)
    << "BIAXIAL STRAIN, TENSION EVAL FAILED WITH ERROR "
    << "'" << Details::get_error_code_string(err) << "'";

  // Biaxial strain, compression
  F(0,0) = 1. - eps; F(0,1) = 0.;       F(0,2) = 0.;
  F(1,0) = 0.;       F(1,1) = 1. - eps; F(1,2) = 0.;
  F(2,0) = 0.;       F(2,1) = 0.;       F(2,2) = 1.;
  err = Details::update(props, rho, F, dtime, temp, T, wave_speed,
                        Fp, ep, epdot, dp, localized);
  EXPECT_TRUE(err == Details::ErrorCode::SUCCESS)
    << "BIAXIAL STRAIN, COMPRESSION EVAL FAILED WITH ERROR "
    << "'" << Details::get_error_code_string(err) << "'";

  // Pure shear, 2D
  F(0,0) = 1.;       F(0,1) = eps;      F(0,2) = 0.;
  F(1,0) = eps;      F(1,1) = 1.;       F(1,2) = 0.;
  F(2,0) = 0.;       F(2,1) = 0.;       F(2,2) = 1.;
  err = Details::update(props, rho, F, dtime, temp, T, wave_speed,
                        Fp, ep, epdot, dp, localized);
  EXPECT_TRUE(err == Details::ErrorCode::SUCCESS)
    << "PURE SHEAR, 2D EVAL FAILED WITH ERROR "
    << "'" << Details::get_error_code_string(err) << "'";

}
} // namespace hyper_ep_utils



#if 0
TEST(HyperEPMaterialModel, ParameterValidation)
{
  using Teuchos::ParameterList;
  scalar_type tol = 1e-14;

  { // Elastic
    auto p = ParameterList("elastic");
    p.set<scalar_type>("E", 10.);
    p.set<scalar_type>("Nu", .1);
    {
      auto params = ParameterList("model");
      params.set("elastic", p);
      Details::Properties props;
      Details::Elastic elastic;
      Details::read_and_validate_elastic_params(params, props, elastic);
      EXPECT_TRUE(std::abs(props.E - 10.) < tol);
      EXPECT_TRUE(std::abs(props.Nu - .1) < tol);
      EXPECT_TRUE(elastic == Details::Elastic::LINEAR_ELASTIC);
    }

    {
      p.set<std::string>("hyperelastic", "neo hookean");
      auto params = ParameterList("model");
      params.set("elastic", p);
      Details::Properties props;
      Details::Elastic elastic;
      Details::read_and_validate_elastic_params(params, props, elastic);
      EXPECT_TRUE(std::abs(props.E - 10.) < tol);
      EXPECT_TRUE(std::abs(props.Nu - .1) < tol);
      EXPECT_TRUE(elastic == Details::Elastic::NEO_HOOKEAN);
    }
  }

  { // Plastic
    auto p0 = ParameterList("plastic");
    p0.set<scalar_type>("A", 10.);
    p0.set<scalar_type>("B", 2.);
    p0.set<scalar_type>("N", .1);
    p0.set<scalar_type>("T0", 400.);
    p0.set<scalar_type>("TM", 500.);
    p0.set<scalar_type>("M", .2);

    auto p1 = ParameterList("rate dependent");
    p1.set<std::string>("type", "johnson cook");
    p1.set<scalar_type>("C", 5.0);
    p1.set<scalar_type>("EPDOT0", 1.0);

    {
      // Von Mises
      auto params = ParameterList("model");
      params.set("plastic", p0);
      Details::Properties props;
      Details::Hardening hardening;
      Details::RateDependence rate_dep;
      Details::read_and_validate_plastic_params(params, props, hardening, rate_dep);
      EXPECT_TRUE(std::abs(props.A - 10.) < tol);
      EXPECT_TRUE(hardening == Details::Hardening::NONE);
      EXPECT_TRUE(rate_dep == Details::RateDependence::NONE);
      EXPECT_TRUE(damage == Details::Damage::NONE);
    }

    {
      // Isotropic hardening
      p0.set<std::string>("hardening", "linear isotropic");
      auto params = ParameterList("model");
      params.set("plastic", p0);
      Details::Properties props;
      Details::Hardening hardening;
      Details::RateDependence rate_dep;
      Details::read_and_validate_plastic_params(params, props, hardening, rate_dep);
      EXPECT_TRUE(std::abs(props.A - 10.) < tol);
      EXPECT_TRUE(std::abs(props.B - 2.) < tol);
      EXPECT_TRUE(hardening == Details::Hardening::LINEAR_ISOTROPIC);
      EXPECT_TRUE(rate_dep == Details::RateDependence::NONE);
    }

    {
      // Power law
      p0.set<std::string>("hardening", "power law");
      auto params = ParameterList("model");
      params.set("plastic", p0);
      Details::Properties props;
      Details::Hardening hardening;
      Details::RateDependence rate_dep;
      Details::read_and_validate_plastic_params(params, props, hardening, rate_dep);
      EXPECT_TRUE(std::abs(props.A - 10.) < tol);
      EXPECT_TRUE(std::abs(props.B - 2.) < tol);
      EXPECT_TRUE(std::abs(props.n - .1) < tol);
      EXPECT_TRUE(hardening == Details::Hardening::POWER_LAW);
      EXPECT_TRUE(rate_dep == Details::RateDependence::NONE);
    }

    {
      // Johnson Cook
      p0.set<std::string>("hardening", "johnson cook");
      auto params = ParameterList("model");
      params.set("plastic", p0);
      Details::Properties props;
      Details::Hardening hardening;
      Details::RateDependence rate_dep;
      Details::read_and_validate_plastic_params(params, props, hardening, rate_dep);
      EXPECT_TRUE(std::abs(props.A - 10.) < tol);
      EXPECT_TRUE(std::abs(props.B - 2.) < tol);
      EXPECT_TRUE(std::abs(props.n - .1) < tol);
      EXPECT_TRUE(std::abs(props.C1 - 400.) < tol);
      EXPECT_TRUE(std::abs(props.C2 - 500.) < tol);
      EXPECT_TRUE(std::abs(props.C3 - .2) < tol);
      EXPECT_TRUE(hardening == Details::Hardening::JOHNSON_COOK);
      EXPECT_TRUE(rate_dep == Details::RateDependence::NONE);
    }

    {
      // Johnson Cook, with rate
      p0.set<std::string>("hardening", "johnson cook");
      p0.set("rate dependent", p1);
      auto params = ParameterList("model");
      params.set("plastic", p0);
      Details::Properties props;
      Details::Hardening hardening;
      Details::RateDependence rate_dep;
      Details::read_and_validate_plastic_params(params, props, hardening, rate_dep);
      EXPECT_TRUE(std::abs(props.A - 10.) < tol);
      EXPECT_TRUE(std::abs(props.B - 2.) < tol);
      EXPECT_TRUE(std::abs(props.n - .1) < tol);
      EXPECT_TRUE(std::abs(props.C1 - 400.) < tol);
      EXPECT_TRUE(std::abs(props.C2 - 500.) < tol);
      EXPECT_TRUE(std::abs(props.C3 - .2) < tol);
      EXPECT_TRUE(hardening == Details::Hardening::JOHNSON_COOK);
      EXPECT_TRUE(rate_dep == Details::RateDependence::JOHNSON_COOK);
    }

    {
      // Zerilli Armstrong, with rate
      auto p_za = ParameterList("plastic");
      p_za.set<std::string>("hardening", "zerilli armstrong");
      p_za.set<scalar_type>("A", 1.);
      p_za.set<scalar_type>("B", 2.);
      p_za.set<scalar_type>("N", 3.);
      p_za.set<scalar_type>("C1", 4.);
      p_za.set<scalar_type>("C2", 5.);
      p_za.set<scalar_type>("C3", 6.);

      auto p_za_r = ParameterList("rate dependent");
      p_za_r.set<std::string>("type", "zerilli armstrong");
      p_za_r.set<double>("C4", 7.);
      p_za.set("rate dependent", p_za_r);

      auto params = ParameterList("model");
      params.set("plastic", p_za);

      Details::Properties props;
      Details::Hardening hardening;
      Details::RateDependence rate_dep;
      Details::read_and_validate_plastic_params(params, props, hardening, rate_dep);
      EXPECT_TRUE(std::abs(props.A - 1.) < tol);
      EXPECT_TRUE(std::abs(props.B - 2.) < tol);
      EXPECT_TRUE(std::abs(props.n - 3.) < tol);
      EXPECT_TRUE(std::abs(props.C1 - 4.) < tol);
      EXPECT_TRUE(std::abs(props.C2 - 5.) < tol);
      EXPECT_TRUE(std::abs(props.C3 - 6.) < tol);
      EXPECT_TRUE(std::abs(props.C4 - 7.) < tol);
      EXPECT_TRUE(hardening == Details::Hardening::ZERILLI_ARMSTRONG);
      EXPECT_TRUE(rate_dep == Details::RateDependence::ZERILLI_ARMSTRONG);
    }
  }
}
#endif


TEST(HyperEPMaterialModel, NeoHookeanHyperElastic)
{
  scalar_type rho = 1.;
  scalar_type temp = 298.;
  scalar_type dtime = 1.;

  Tensor<3> T;
  Tensor<3> F = identity_tensor<3>();
  Tensor<3> Fp = identity_tensor<3>();

  Details::Properties props;
  props.elastic = Details::Elastic::NEO_HOOKEAN;
  props.E = 10.;
  props.Nu = .1;
  props.A = std::numeric_limits<scalar_type>::max();

  scalar_type C10 = props.E / (4. * (1. + props.Nu));
  scalar_type D1 = 6. * (1. - 2. * props.Nu) / props.E;

  // Uniaxial strain
  scalar_type f1 = 1.1;
  F(0,0) = f1;
  scalar_type c = 0.;
  scalar_type ep = 0.;
  scalar_type epdot = 0.;
  scalar_type dp = 0.;
  scalar_type localized = 0.;
  Details::update(props, rho, F, dtime, temp, T, c,
                  Fp, ep, epdot, dp, localized);

  scalar_type fac = 1.66666666666667;  // 10/6
  scalar_type sxx = (2.0/3.0)*std::pow(f1,-fac)*(-2.*C10*D1*(-f1*f1+1)+3*std::pow(f1,fac)*(f1-1))/D1;
  scalar_type syy = (2.0/3.0)*std::pow(f1,-fac)*(C10*D1*(-f1*f1+1)+3*std::pow(f1,fac)*(f1-1))/D1;

  EXPECT_TRUE(Omega_h::are_close(T(0,0), sxx, 5e-7));
  EXPECT_TRUE(Omega_h::are_close(T(1,1), syy, 5e-7));
  EXPECT_TRUE(Omega_h::are_close(T(1,1), T(2,2)));
  EXPECT_TRUE(Omega_h::are_close(T(0,1), 0.0));
  EXPECT_TRUE(Omega_h::are_close(ep, 0.0));
}


TEST(HyperEPMaterialModel, LinearElastic)
{
  scalar_type rho = 1.;
  scalar_type temp = 298.;
  scalar_type dtime = 1.;

  Tensor<3> T;
  Tensor<3> F = identity_tensor<3>();
  Tensor<3> Fp = identity_tensor<3>();

  Details::Properties props;
  props.elastic = Details::Elastic::LINEAR_ELASTIC;
  props.E = 10.;
  props.Nu = .1;
  props.A = std::numeric_limits<scalar_type>::max();

  // Uniaxial strain
  scalar_type eps = .1;
  F(0,0) = 1.0 + eps;
  scalar_type c = 0.;
  scalar_type ep = 0.;
  scalar_type epdot = 0.;
  scalar_type dp = 0.;
  scalar_type localized = 0.;
  Details::update(props, rho, F, dtime, temp, T, c,
                  Fp, ep, epdot, dp, localized);

  scalar_type K = props.E / 3. / (1. - 2. * props.Nu);
  scalar_type G = props.E / 2. / (1. + props.Nu);
  scalar_type sxx = K * eps + 4.0 / 3.0 * G * eps;
  scalar_type syy = K * eps - 2.0 / 3.0 * G * eps;
  EXPECT_TRUE(Omega_h::are_close(T(0,0), sxx));
  EXPECT_TRUE(Omega_h::are_close(T(1,1), syy));
  EXPECT_TRUE(Omega_h::are_close(T(1,1), T(2,2)));
  EXPECT_TRUE(Omega_h::are_close(T(0,1), 0.0));
  EXPECT_TRUE(Omega_h::are_close(ep, 0.0));
}


TEST(HyperEPMaterialModel, SimpleJ2)
{
  scalar_type rho = 1.;
  scalar_type temp = 298.;
  scalar_type dtime = 1.;

  Tensor<3> T;
  Tensor<3> F = identity_tensor<3>();
  Tensor<3> Fp = identity_tensor<3>();

  Details::Properties props;
  props.elastic = Details::Elastic::NEO_HOOKEAN;
  props.E = 10e6;
  props.Nu = 0.1;
  props.A = 40e3;

  scalar_type c = 0.;
  scalar_type ep = 0.;
  scalar_type epdot = 0.;
  scalar_type dp = 0.;
  scalar_type localized = 0.;
  // Uniaxial strain
  F(0,0) = 1.004;
  Details::update(props, rho, F, dtime, temp, T, c,
                  Fp, ep, epdot, dp, localized);
  EXPECT_TRUE(Omega_h::are_close(ep, 0.));

  c = 0.;
  F(0,0) = 1.005;
  Details::update(props, rho, F, dtime, temp, T, c,
                  Fp, ep, epdot, dp, localized);
  EXPECT_TRUE(Omega_h::are_close(T(0,0), 47500.));
  EXPECT_TRUE(Omega_h::are_close(T(1,1), 7500.));
  EXPECT_TRUE(Omega_h::are_close(T(1,1), T(2,2)));
  EXPECT_TRUE(Omega_h::are_close(ep, 0.00076134126264676861, 1e-6));
}


TEST(HyperEPMaterialModel, NonHardeningRadialReturn)
{
  scalar_type temp = 298.;
  scalar_type dtime = 1.;

  Tensor<3> T;
  Tensor<3> F = identity_tensor<3>();
  Tensor<3> Fp = identity_tensor<3>();

  Details::Properties props;
  props.elastic = Details::Elastic::NEO_HOOKEAN;
  props.E = 10e6;
  props.Nu = 0.1;
  props.A = 40e3;

  scalar_type ep = 0.;
  scalar_type epdot = 0.;
  scalar_type dp = 0.;

  Details::StateFlag flag;

  // Uniaxial stress, below yield
  Tensor<3> Te = zero_matrix<3, 3>();
  scalar_type fac = .9;
  Te(0,0) = fac * props.A;
  flag = Details::StateFlag::TRIAL;
  Details::radial_return(props, Te, F, temp, dtime, T, Fp, ep, epdot, dp, flag);

  EXPECT_TRUE(Omega_h::are_close(T(0,0), Te(0,0)));
  EXPECT_TRUE(Omega_h::are_close(T(1,1), Te(1,1)));
  EXPECT_TRUE(Omega_h::are_close(T(2,2), Te(2,2)));
  EXPECT_TRUE(Omega_h::are_close(T(0,1), 0.));
  EXPECT_TRUE(Omega_h::are_close(T(0,2), 0.));
  EXPECT_TRUE(Omega_h::are_close(T(1,2), 0.));

  // Uniaxial stress, above yield
  fac = 1.1;
  Te(0,0) = fac * props.A;
  flag = Details::StateFlag::TRIAL;
  Details::radial_return(props, Te, F, temp, dtime, T, Fp, ep, epdot, dp, flag);

  scalar_type Txx = 2.*std::pow(props.A,2)*fac/(3.*props.A*fac) + props.A*fac/3.;
  scalar_type Tyy =   -std::pow(props.A,2)*fac/(3.*props.A*fac) + props.A*fac/3.;
  EXPECT_TRUE(Omega_h::are_close(T(0,0), Txx));
  EXPECT_TRUE(Omega_h::are_close(T(1,1), Tyy));
  EXPECT_TRUE(Omega_h::are_close(T(0,1), 0.));
  EXPECT_TRUE(Omega_h::are_close(T(0,2), 0.));
  EXPECT_TRUE(Omega_h::are_close(T(1,2), 0.));
}


TEST(HyperEPMaterialModel, JohnsonCookDamage2)
{
  // Material properties
  auto rho = hyper_ep_utils::copper_density();
  auto props = hyper_ep_utils::copper_johnson_cook_props();
  props.allow_no_shear = true;
  props.allow_no_tension = false;
  props.set_stress_to_zero = false;

  // Initialize in/out variables to be updated by material update
  Tensor<3> F = identity_tensor<3>();
  Tensor<3> Fp = identity_tensor<3>();
  auto T = 0.0 * F;

  scalar_type dtime = 1.;
  scalar_type temp = 298.;
  scalar_type wave_speed = 0.;
  scalar_type ep = 0.;
  scalar_type epdot = 0.;
  scalar_type dp = 0.;
  scalar_type localized = 0.;

  Details::ErrorCode err;

  auto n = 100;
  scalar_type eps = 0.1;
  for (int i=0; i<n; i++) {
    auto fac = static_cast<scalar_type>(i+1) / static_cast<scalar_type>(n);
    auto e = fac * eps;

    // Simple shear
    F(0,0) = 1.;       F(0,1) = e;        F(0,2) = 0.;
    F(1,0) = 0.;       F(1,1) = 1.;       F(1,2) = 0.;
    F(2,0) = 0.;       F(2,1) = 0.;       F(2,2) = 1.;
    err = Details::update(props, rho, F, dtime, temp, T, wave_speed,
                          Fp, ep, epdot, dp, localized);
    EXPECT_TRUE(err == Details::ErrorCode::SUCCESS)
      << "SIMPLE SHEAR, DAMAGE, 2D EVAL FAILED WITH ERROR "
      << "'" << Details::get_error_code_string(err) << "'";
    std::cout << "LOCALIZED: " << localized << std::endl;
    std::cout << "DP = " << dp << std::endl;
    std::cout << "FP = [";
    for (int j=0; j<3; j++)
      for (int k=0; k<3; k++)
        std::cout << Fp(j,k) << " ";
    std::cout << "]" << std::endl;
  }
}


// The following are not actually tests. They run different combinations of
// model behaviors through several prescribed motions. But, results are not
// checked, other than to check that the motion did not throw.
TEST(HyperEPMaterialModel, ElasticPerfectlyPlasticMotionNoTest)
{
  scalar_type eps = 0.01;
  auto rho = hyper_ep_utils::copper_density();
  auto props = hyper_ep_utils::copper_johnson_cook_props();
  props.hardening = Details::Hardening::NONE;
  props.rate_dep = Details::RateDependence::NONE;
  props.damage = Details::Damage::NONE;
  hyper_ep_utils::eval_prescribed_motions(eps, props, rho);
}


TEST(HyperEPMaterialModel, LinearIsotropicHardeningMotionNoTest)
{
  scalar_type eps = 0.01;
  auto rho = hyper_ep_utils::copper_density();
  auto props = hyper_ep_utils::copper_johnson_cook_props();
  props.hardening = Details::Hardening::LINEAR_ISOTROPIC;
  props.rate_dep = Details::RateDependence::NONE;
  props.damage = Details::Damage::NONE;
  hyper_ep_utils::eval_prescribed_motions(eps, props, rho);
}


TEST(HyperEPMaterialModel, PowerLawHardeningMotionNoTest)
{
  scalar_type eps = 0.01;
  auto rho = hyper_ep_utils::copper_density();
  auto props = hyper_ep_utils::copper_johnson_cook_props();
  props.hardening = Details::Hardening::POWER_LAW;
  props.rate_dep = Details::RateDependence::NONE;
  props.damage = Details::Damage::NONE;
  hyper_ep_utils::eval_prescribed_motions(eps, props, rho);
}


TEST(HyperEPMaterialModel, JohnsonCookHardeningMotionNoTest)
{
  scalar_type eps = 0.01;
  auto rho = hyper_ep_utils::copper_density();
  auto props = hyper_ep_utils::copper_johnson_cook_props();
  props.rate_dep = Details::RateDependence::NONE;
  props.damage = Details::Damage::NONE;
  hyper_ep_utils::eval_prescribed_motions(eps, props, rho);
}


TEST(HyperEPMaterialModel, JohnsonCookRateDependentMotionNoTest)
{
  scalar_type eps = 0.01;
  auto rho = hyper_ep_utils::copper_density();
  auto props = hyper_ep_utils::copper_johnson_cook_props();
  props.damage = Details::Damage::NONE;
  hyper_ep_utils::eval_prescribed_motions(eps, props, rho);
}


TEST(HyperEPMaterialModel, JohnsonCookDamageMotionNoTest)
{
  scalar_type eps = 0.01;
  auto rho = hyper_ep_utils::copper_density();
  auto props = hyper_ep_utils::copper_johnson_cook_props();
  hyper_ep_utils::eval_prescribed_motions(eps, props, rho);
}


TEST(HyperEPMaterialModel, ZerilliArmstrongHardeningMotionNoTest)
{
  scalar_type eps = 0.01;
  auto rho = hyper_ep_utils::copper_density();
  auto props = hyper_ep_utils::copper_zerilli_armstrong_props();
  props.rate_dep = Details::RateDependence::NONE;
  hyper_ep_utils::eval_prescribed_motions(eps, props, rho);
}


TEST(HyperEPMaterialModel, ZerilliArmstrongRateDependentMotionNoTest)
{
  scalar_type eps = 0.01;
  auto rho = hyper_ep_utils::copper_density();
  auto props = hyper_ep_utils::copper_zerilli_armstrong_props();
  hyper_ep_utils::eval_prescribed_motions(eps, props, rho);
}

} // namespace

#endif
