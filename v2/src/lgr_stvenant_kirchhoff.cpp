#include <lgr_for.hpp>
#include <lgr_simulation.hpp>
#include <lgr_stvenant_kirchhoff.hpp>

namespace lgr {

template <class Elem>
struct StVenantKirchhoff : public Model<Elem> {
  FieldIndex bulk_modulus;
  FieldIndex shear_modulus;
  FieldIndex deformation_gradient;
  StVenantKirchhoff(Simulation& sim_in, Omega_h::InputMap& pl)
      : Model<Elem>(sim_in, pl) {
    this->bulk_modulus = this->point_define(
        "kappa", "bulk modulus", 1, RemapType::PER_UNIT_VOLUME, pl, "");
    this->shear_modulus = this->point_define(
        "mu", "shear modulus", 1, RemapType::PER_UNIT_VOLUME, pl, "");
    constexpr auto dim = Elem::dim;
    this->deformation_gradient = this->point_define("F", "deformation gradient",
        square(dim), RemapType::POLAR, pl, "I");
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
        for (int j = 0; j < Elem::dim; ++j) F(i, j) = F_small(i, j);
      Matrix<3, 3> sigma;
      double c;
      stvenant_kirchhoff_update(kappa, nu, rho, F, sigma, c);
      setstress(points_to_stress, point, sigma);
      points_to_wave_speed[point] = c;
    };
    parallel_for(
        "StVenant-Kirchhoff kernel", this->points(), std::move(functor));
  }
};

template <class Elem>
ModelBase* stvenant_kirchhoff_factory(
    Simulation& sim, std::string const&, Omega_h::InputMap& pl) {
  return new StVenantKirchhoff<Elem>(sim, pl);
}

#define LGR_EXPL_INST(Elem)                                                    \
  template ModelBase* stvenant_kirchhoff_factory<Elem>(                        \
      Simulation&, std::string const&, Omega_h::InputMap&);
LGR_EXPL_INST_ELEMS
#undef LGR_EXPL_INST

}  // namespace lgr
