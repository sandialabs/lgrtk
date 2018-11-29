#include <lgr_stvenant_kirchhoff.hpp>
#include <lgr_simulation.hpp>
#include <lgr_for.hpp>

namespace lgr {

OMEGA_H_INLINE void stvenant_kirchhoff_update(
    double bulk_modulus,
    double shear_modulus,
    double density,
    Matrix<3, 3> F,
    Matrix<3, 3>& stress,
    double& wave_speed) {
  OMEGA_H_CHECK(density > 0.0);
  auto const J = Omega_h::determinant(F);
  OMEGA_H_CHECK(J > 0.0);
  auto const Jinv = 1.0 / J;
  auto const I = Omega_h::identity_matrix<3, 3>();
  auto const C = transpose(F) * F;
  auto const E = 0.5 * (C - I);
  auto const mu = shear_modulus;
  auto const lambda = bulk_modulus - 2.0 * mu / 3.0;
  auto const S = lambda * trace(E) * I + 2.0 * mu * E;
  stress = Jinv * F * S * transpose(F);
  wave_speed = std::sqrt(bulk_modulus / density);
  OMEGA_H_CHECK(wave_speed > 0.0);
}

template <class Elem>
struct StVenantKirchhoff : public Model<Elem> {
  FieldIndex bulk_modulus;
  FieldIndex shear_modulus;
  FieldIndex deformation_gradient;
  StVenantKirchhoff(Simulation& sim_in, Omega_h::InputMap& pl):Model<Elem>(sim_in, pl) {
    this->bulk_modulus =
      this->point_define("kappa", "bulk modulus", 1,
          RemapType::PER_UNIT_VOLUME, pl, "");
    this->shear_modulus =
      this->point_define("mu", "shear modulus", 1,
        RemapType::PER_UNIT_VOLUME, pl, "");
    constexpr auto dim = Elem::dim;
    this->deformation_gradient =
      this->point_define("F", "deformation gradient",
          square(dim), RemapType::POSITIVE_DETERMINANT, pl, "I");
  }
  std::uint64_t exec_stages() override final { return AT_MATERIAL_MODEL; }
  char const* name() override final { return "StVenant-Kirchhoff"; }
  void at_material_model() override final {
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
      stvenant_kirchhoff_update(kappa, nu, rho, F, sigma, c);
      setsymm<Elem>(points_to_stress, point, resize<Elem::dim>(sigma));
      points_to_wave_speed[point] = c;
    };
    parallel_for("StVenant-Kirchhoff kernel", this->points(), std::move(functor));
  }
};

template <class Elem>
ModelBase* stvenant_kirchhoff_factory(Simulation& sim, std::string const&, Omega_h::InputMap& pl) {
  return new StVenantKirchhoff<Elem>(sim, pl);
}

#define LGR_EXPL_INST(Elem) \
template ModelBase* stvenant_kirchhoff_factory<Elem>(Simulation&, std::string const&, Omega_h::InputMap&);
LGR_EXPL_INST_ELEMS
#undef LGR_EXPL_INST

}
