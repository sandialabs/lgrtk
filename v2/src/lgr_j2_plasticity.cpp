#include <lgr_for.hpp>
#include <lgr_j2_plasticity.hpp>
#include <lgr_simulation.hpp>

namespace lgr {

enum class Hardening
{
  POWER_LAW, VOCE
};

enum class RateSensitivity
{
  POWER_LAW, ARCSINH
};

struct Properties
{
  // Elasticity
  double E{0.0};
  double nu{0.0};
  double kappa{0.0};
  double mu{0.0};

  // Plasticity

  Hardening hardening{Hardening::POWER_LAW};

  // virgin yield strength
  double Y0{0.0};

  // power law
  double n{0.0};
  double eps0{0.0};

  // Voce
  double Ysat{0.0};
  double H0{0.0};

  RateSensitivity rate_sensitivity{RateSensitivity::POWER_LAW};

  // reference stress
  double S0{0.0};

  // arcsinh
  double epsdot0{0.0};

  // power law, also uses epsdot0
  double m{0.0};
};

void read_and_validate_elastic_params(
    Omega_h::InputMap & pl, Properties & props)
{
  if (!pl.is<double>("E")) {
    Omega_h_fail("Young's modulus \"E\" modulus must be defined");
  }
  double const E = pl.get<double>("E", 0.0);
  if (E <= 0.0) {
    Omega_h_fail("Young's modulus \"E\" must be positive");
  }
  if (!pl.is<double>("nu")) {
    Omega_h_fail("Poisson's ratio \"nu\" must be defined");
  }
  double const nu = pl.get<double>("nu", 0.0);
  if (nu <= -1.0 || nu >= 0.5) {
    Omega_h_fail("Invalid value for Poisson's ratio \"nu\"");
  }
  props.E = E;
  props.nu = nu;
  props.kappa = E / 3.0 / (1.0 - 2.0 * nu);
  props.mu = E / 2.0 / (1.0 + nu);
}

void read_and_validate_hardening_params(
    Omega_h::InputMap & pl, Properties & props)
{
  if (!pl.is<std::string>("Hardening")) {
    Omega_h_fail("Hardening law must be defined");
  }
  std::string const hardening_str = pl.get<std::string>("Hardening", "NONE");
  if (hardening_str == "Power Law") {
    double const Y0 = pl.get<double>("Y0", 0.0);
    if (Y0 <= 0.0) {
      Omega_h_fail("Virgin yield strength \"Y0\" must be positive");
    }
    double const n = pl.get<double>("n", 0.0);
    if (n < 0.0) {
      Omega_h_fail("Hardening exponent \"n\" must be non-negative");
    }
    double const eps0 = pl.get<double>("eps0", 0.0);
    if (eps0 <= 0.0) {
      Omega_h_fail("Reference plastic strain \"eps0\" must be positive");
    }
    props.hardening = Hardening::POWER_LAW;
    props.Y0 = Y0;
    props.n = n;
    props.eps0 = eps0;
  } else if (hardening_str == "Voce") {
    double const Y0 = pl.get<double>("Y0", 0.0);
    if (Y0 <= 0.0) {
      Omega_h_fail("Virgin yield strength \"Y0\" must be positive");
    }
    double const Ysat = pl.get<double>("Ysat", 0.0);
    if (Ysat <= 0.0) {
      Omega_h_fail("Saturation strength \"n\" must be positive");
    }
    double const H0 = pl.get<double>("H0", 0.0);
    if (H0 < 0.0) {
      Omega_h_fail("Hardening modulus \"eps0\" must be non-negative");
    }
    props.hardening = Hardening::VOCE;
    props.Y0 = Y0;
    props.Ysat = Ysat;
    props.H0 = H0;
  } else {
    Omega_h_fail("Unrecognized hardening type");
  }
  return;
}

void read_and_validate_rate_sensitivity_params(
    Omega_h::InputMap & pl, Properties & props)
{
  if (!pl.is<std::string>("Rate Sensitivity")) {
    Omega_h_fail("Rate Sensitivity law must be defined");
  }
  std::string const rate_str = pl.get<std::string>("Rate Sensitivity", "NONE");
  if (rate_str == "Power Law") {
    double const S0 = pl.get<double>("S0", 0.0);
    if (S0 <= 0.0) {
      Omega_h_fail("Reference stress \"S0\" must be positive");
    }
    double const m = pl.get<double>("m", 0.0);
    if (m < 0.0) {
      Omega_h_fail("Rate sensitivity exponent \"n\" must be positive");
    }
    double const epsdot0 = pl.get<double>("epsdot0", 0.0);
    if (epsdot0 <= 0.0) {
      Omega_h_fail("Reference plastic strain rate \"epsdot0\" must be positive");
    }
    props.hardening = Hardening::POWER_LAW;
    props.S0 = S0;
    props.m = m;
    props.epsdot0 = epsdot0;
  } else if (rate_str == "Arcsinh") {
    double const S0 = pl.get<double>("S0", 0.0);
    if (S0 <= 0.0) {
      Omega_h_fail("Reference stress \"S0\" must be positive");
    }
    double const epsdot0 = pl.get<double>("epsdot0", 0.0);
    if (epsdot0 <= 0.0) {
      Omega_h_fail("Reference plastic strain rate \"epsdot0\" must be positive");
    }
    props.hardening = Hardening::POWER_LAW;
    props.S0 = S0;
    props.epsdot0 = epsdot0;
  } else {
    Omega_h_fail("Unrecognized hardening type");
  }
  return;
}

OMEGA_H_INLINE void j2_plasticity_update(double bulk_modulus,
    double shear_modulus, double density, Tensor<3> F, Tensor<3>& stress,
    double& wave_speed)
{
  OMEGA_H_CHECK(density > 0.0);
  auto const J = Omega_h::determinant(F);
  OMEGA_H_CHECK(J > 0.0);
  auto const Jinv = 1.0 / J;
  auto const half_bulk_modulus = (1.0 / 2.0) * bulk_modulus;
  auto const volumetric_stress = half_bulk_modulus * (J - Jinv);
  auto const I = Omega_h::identity_matrix<3, 3>();
  auto const isotropic_stress = volumetric_stress * I;
  auto const Jm13 = 1.0 / std::cbrt(J);
  auto const Jm23 = Jm13 * Jm13;
  auto const Jm53 = Jm23 * Jm23 * Jm13;
  auto const B = F * transpose(F);
  auto const devB = Omega_h::deviator(B);
  auto const deviatoric_stress = shear_modulus * Jm53 * devB;
  stress = isotropic_stress + deviatoric_stress;
  auto const tangent_bulk_modulus = half_bulk_modulus * (J + Jinv);
  auto const plane_wave_modulus =
      tangent_bulk_modulus + (4.0 / 3.0) * shear_modulus;
  OMEGA_H_CHECK(plane_wave_modulus > 0.0);
  wave_speed = std::sqrt(plane_wave_modulus / density);
  OMEGA_H_CHECK(wave_speed > 0.0);
}

template<class Elem>
struct J2Plasticity: public Model<Elem>
{
  FieldIndex bulk_modulus;
  FieldIndex shear_modulus;
  FieldIndex deformation_gradient;
  J2Plasticity(Simulation& sim_in, Omega_h::InputMap& pl)
  :
      Model<Elem>(sim_in, pl)
  {
    this->bulk_modulus = this->point_define(
        "kappa", "bulk modulus", 1, RemapType::PER_UNIT_VOLUME, pl, "");
    this->shear_modulus = this->point_define(
        "mu", "shear modulus", 1, RemapType::PER_UNIT_VOLUME, pl, "");
    constexpr auto dim = Elem::dim;
    this->deformation_gradient = this->point_define(
        "F", "deformation gradient", square(dim), RemapType::POLAR, pl, "I");
  }
  std::uint64_t exec_stages() override final
  {
    return AT_MATERIAL_MODEL;
  }
  char const* name() override final
  {
    return "neo-Hookean";
  }
  void at_material_model() override final
  {
    auto points_to_kappa = this->points_get(this->bulk_modulus);
    auto points_to_nu = this->points_get(this->shear_modulus);
    auto points_to_rho = this->points_get(this->sim.density);
    auto points_to_F = this->points_get(this->deformation_gradient);
    auto points_to_stress = this->points_set(this->sim.stress);
    auto points_to_wave_speed = this->points_set(this->sim.wave_speed);
    auto functor = OMEGA_H_LAMBDA(int point) {
      auto F_small = getfull<Elem>(points_to_F, point);
      auto kappa = points_to_kappa[point];
      auto nu = points_to_nu[point];
      auto rho = points_to_rho[point];
      auto F = identity_tensor<3>();
      for (int i = 0; i < Elem::dim; ++i) {
        for (int j = 0; j < Elem::dim; ++j) {
          F(i, j) = F_small(i, j);
        }
      }
      Tensor<3> sigma;
      double c;
      j2_plasticity_update(kappa, nu, rho, F, sigma, c);
      setstress(points_to_stress, point, sigma);
      points_to_wave_speed[point] = c;
    };
    parallel_for(this->points(), std::move(functor));
  }
};

template<class Elem>
ModelBase* j2_plasticity_factory(
    Simulation& sim, std::string const&, Omega_h::InputMap& pl)
{
  return new J2Plasticity<Elem>(sim, pl);
}

#define LGR_EXPL_INST(Elem)                                                    \
		template ModelBase* j2_plasticity_factory<Elem>(                               \
				Simulation&, std::string const&, Omega_h::InputMap&);
LGR_EXPL_INST_ELEMS
#undef LGR_EXPL_INST

}
 // namespace lgr
