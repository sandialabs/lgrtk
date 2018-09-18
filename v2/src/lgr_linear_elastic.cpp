#include <lgr_linear_elastic.hpp>
#include <lgr_simulation.hpp>
#include <lgr_for.hpp>

namespace lgr {

OMEGA_H_INLINE void linear_elastic_update(
    double bulk_modulus,
    double shear_modulus,
    double density,
    Matrix<3, 3> grad_u,
    Matrix<3, 3>& stress,
    double& wave_speed) {
  auto strain = (1.0 / 2.0) * (grad_u + transpose(grad_u));
  auto I = identity_matrix<3, 3>();
  auto isotropic_strain = (trace(strain) / 3.) * I;
  auto deviatoric_strain = strain - isotropic_strain;
  stress = (3. * bulk_modulus) * isotropic_strain + (2.0 * shear_modulus) * deviatoric_strain;
  auto plane_wave_modulus =
    bulk_modulus + (4.0 / 3.0) * shear_modulus;
  wave_speed = std::sqrt( plane_wave_modulus / density );
}

template <class Elem>
struct LinearElastic : public Model<Elem> {
  FieldIndex bulk_modulus;
  FieldIndex shear_modulus;
  FieldIndex deformation_gradient;
  LinearElastic(Simulation& sim_in, Teuchos::ParameterList& pl):Model<Elem>(sim_in, pl) {
    this->bulk_modulus =
      this->point_define("kappa", "bulk modulus", 1,
          RemapType::PER_UNIT_VOLUME, "");
    this->shear_modulus =
      this->point_define("mu", "shear modulus", 1,
        RemapType::PER_UNIT_VOLUME, "");
    constexpr auto dim = Elem::dim;
    this->deformation_gradient =
      this->point_define("F", "deformation gradient",
          square(dim), RemapType::POSITIVE_DETERMINANT, "I");
  }
  ModelOrder order() override final { return IS_MATERIAL_MODEL; }
  char const* name() override final { return "linear elastic"; }
  void update_state() override final {
    auto points_to_kappa = this->points_get(this->bulk_modulus);
    auto points_to_nu = this->points_get(this->shear_modulus);
    auto points_to_rho = this->points_get(this->sim.density);
    auto points_to_F = this->points_get(this->deformation_gradient);
    auto points_to_stress = this->points_set(this->sim.stress);
    auto points_to_wave_speed = this->points_set(this->sim.wave_speed);
    auto functor = OMEGA_H_LAMBDA(int point) {
      auto F = getfull<Elem>(points_to_F, point);
      auto kappa = points_to_kappa[point];
      auto nu = points_to_nu[point];
      auto rho = points_to_rho[point];
      auto I = identity_matrix<Elem::dim, Elem::dim>();
      auto grad_u = resize<3>(F - I);
      Matrix<3, 3> sigma;
      double c;
      linear_elastic_update(kappa, nu, rho, grad_u, sigma, c);
      setsymm<Elem>(points_to_stress, point, resize<Elem::dim>(sigma));
      points_to_wave_speed[point] = c;
    };
    parallel_for("linear elastic kernel", this->points(), std::move(functor));
  }
};

template <class Elem>
ModelBase* linear_elastic_factory(Simulation& sim, std::string const&, Teuchos::ParameterList& pl) {
  return new LinearElastic<Elem>(sim, pl);
}

#define LGR_EXPL_INST(Elem) \
template ModelBase* linear_elastic_factory<Elem>(Simulation&, std::string const&, Teuchos::ParameterList&);
LGR_EXPL_INST_ELEMS
#undef LGR_EXPL_INST

}
