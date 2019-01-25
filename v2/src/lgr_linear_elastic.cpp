#include <lgr_for.hpp>
#include <lgr_linear_elastic.hpp>
#include <lgr_simulation.hpp>

namespace lgr {

OMEGA_H_INLINE void linear_elastic_update(double bulk_modulus,
    double shear_modulus, double density, Matrix<3, 3> grad_u,
    Matrix<3, 3>& stress, double& wave_speed) {
  auto strain = (1.0 / 2.0) * (grad_u + transpose(grad_u));
  auto I = identity_matrix<3, 3>();
  auto isotropic_strain = (trace(strain) / 3.) * I;
  auto deviatoric_strain = strain - isotropic_strain;
  stress = (3. * bulk_modulus) * isotropic_strain +
           (2.0 * shear_modulus) * deviatoric_strain;
  auto plane_wave_modulus = bulk_modulus + (4.0 / 3.0) * shear_modulus;
  wave_speed = std::sqrt(plane_wave_modulus / density);
}

template <class Elem>
struct LinearElastic : public Model<Elem> {
  FieldIndex bulk_modulus;
  FieldIndex shear_modulus;
  FieldIndex deformation_gradient;
  FieldIndex effective_bulk_modulus;
  LinearElastic(Simulation& sim_in, Omega_h::InputMap& pl)
      : Model<Elem>(sim_in, pl) {
    this->bulk_modulus = this->point_define(
        "kappa", "bulk modulus", 1, RemapType::PER_UNIT_VOLUME, pl, "");
    this->shear_modulus = this->point_define(
        "mu", "shear modulus", 1, RemapType::PER_UNIT_VOLUME, pl, "");
    constexpr auto dim = Elem::dim;
    this->deformation_gradient = this->point_define("F", "deformation gradient",
        square(dim), RemapType::POLAR, pl, "I");
    this->effective_bulk_modulus = this->point_define("kappa_tilde",
        "effective bulk modulus", 1, RemapType::PER_UNIT_VOLUME, pl, "");
  }
  std::uint64_t exec_stages() override final { return AT_MATERIAL_MODEL; }
  char const* name() override final { return "linear elastic"; }
  void at_material_model() override final {
    auto points_to_kappa = this->points_get(this->bulk_modulus);
    auto points_to_kappa_tilde = this->points_set(this->effective_bulk_modulus);
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
      setstress(points_to_stress, point, sigma);
      points_to_wave_speed[point] = c;
      points_to_kappa_tilde[point] = kappa;
    };
    parallel_for(this->points(), std::move(functor));
  }
};

template <class Elem>
ModelBase* linear_elastic_factory(
    Simulation& sim, std::string const&, Omega_h::InputMap& pl) {
  return new LinearElastic<Elem>(sim, pl);
}

#define LGR_EXPL_INST(Elem)                                                    \
  template ModelBase* linear_elastic_factory<Elem>(                            \
      Simulation&, std::string const&, Omega_h::InputMap&);
LGR_EXPL_INST_ELEMS
#undef LGR_EXPL_INST

}  // namespace lgr
