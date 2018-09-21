#include <lgr_neo_hookean.hpp>
#include <lgr_simulation.hpp>
#include <lgr_for.hpp>

namespace lgr {

OMEGA_H_INLINE void neo_hookean_update(
    double bulk_modulus,
    double shear_modulus,
    double density,
    Matrix<3, 3> F,
    Matrix<3, 3>& stress,
    double& wave_speed) {
  auto const B = F * transpose(F);
  auto const J = Omega_h::determinant(F);
  auto const Jinv = 1.0 / J;
  auto const half_bulk_modulus = (1.0 / 2.0) * bulk_modulus;
  auto const negative_pressure = half_bulk_modulus * (J - Jinv);
  auto const I = Omega_h::identity_matrix<3, 3>();
  auto const volumetric_stress = negative_pressure * I;
  auto const Jm13 = 1.0 / std::cbrt(J);
  auto const Jm23 = Jm13 * Jm13;
  auto const Jm53 = Jm23 * Jm23 * Jm13;
  auto const devB = Omega_h::deviator(B);
  auto const deviatoric_stress = shear_modulus * Jm53 * devB;
  stress = volumetric_stress + deviatoric_stress;
  auto const tangent_bulk_modulus = half_bulk_modulus * (J + Jinv);
  auto const plane_wave_modulus =
    tangent_bulk_modulus + (4.0 / 3.0) * shear_modulus;
  wave_speed = std::sqrt(plane_wave_modulus / density);
}

template <class Elem>
struct NeoHookean : public Model<Elem> {
  FieldIndex bulk_modulus;
  FieldIndex shear_modulus;
  FieldIndex deformation_gradient;
  NeoHookean(Simulation& sim_in, Teuchos::ParameterList& pl):Model<Elem>(sim_in, pl) {
    this->bulk_modulus =
      this->point_define("kappa", "bulk modulus", 1,
          RemapType::PER_UNIT_VOLUME, "");
    this->shear_modulus =
      this->point_define("mu", "shear modulus", 1,
        RemapType::PER_UNIT_VOLUME, "");
    constexpr auto dim = Elem::dim;
    this->deformation_gradient =
      this->point_define("F", "deformation gradient",
          square(dim), RemapType::NONE, "I");
  }
  ModelOrder order() override final { return IS_MATERIAL_MODEL; }
  char const* name() override final { return "neo-Hookean"; }
  void update_state() override final {
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
      auto F = identity_matrix<3, 3>();
      for (int i = 0; i < Elem::dim; ++i)
      for (int j = 0; j < Elem::dim; ++j)
        F(i,j) = F_small(i,j);
      Matrix<3, 3> sigma;
      double c;
      neo_hookean_update(kappa, nu, rho, F, sigma, c);
      setsymm<Elem>(points_to_stress, point, resize<Elem::dim>(sigma));
      points_to_wave_speed[point] = c;
    };
    parallel_for("neo-Hookean kernel", this->points(), std::move(functor));
  }
};

template <class Elem>
ModelBase* neo_hookean_factory(Simulation& sim, std::string const&, Teuchos::ParameterList& pl) {
  return new NeoHookean<Elem>(sim, pl);
}

#define LGR_EXPL_INST(Elem) \
template ModelBase* neo_hookean_factory<Elem>(Simulation&, std::string const&, Teuchos::ParameterList&);
LGR_EXPL_INST_ELEMS
#undef LGR_EXPL_INST

}
