#include <lgr_albany_J2.hpp>
#include <lgr_exp.hpp>
#include <lgr_for.hpp>
#include <lgr_simulation.hpp>

namespace lgr {

OMEGA_H_INLINE void albany_J2_update(
    double const rho,
    double const E,
    double const nu,
    double const K,
    double const Y,
    Tensor<3> const& F,
    Tensor<3>& Fp,
    double& eqps,
    Tensor<3>& sigma,
    double& c){

  auto const I = Omega_h::identity_matrix<3, 3>();
  double const sqrt23 = std::sqrt(2.0 / 3.0);

  // compute material properties
  auto const mu = E / (2.0 * (1.0 + nu));
  auto const kappa = E / (3.0 * (1.0 - 2.0 * nu));

  // get def grad quantities
  auto const J = determinant(F);
  auto const Jp13 = std::cbrt(J);
  auto const Jm23 = 1.0 / Jp13 / Jp13;
  auto const Fpinv = invert(Fp);

  // compute the trial state
  auto const Cpinv = Fpinv * transpose(Fpinv);
  auto const be = Jm23 * F * Cpinv * transpose(F);
  auto const mubar = trace(be) * mu / 3.0;
  auto s = mu * deviator(be);

  // check the yield condition
  auto const smag = Omega_h::norm(s);
  auto const f = smag - sqrt23 * (Y + K * eqps);

  // plastic increment
  if (f > 1.0e-12) {

    int iter = 0;
    bool converged = false;
    double dgam{0.0}, H{0.0}, dH{0.0}, alpha{0.0}, res{0.0};

    double X = 0.;
    double R = f;
    double dRdX = -2.0 * mubar * (1.0 + H / (3.0 * mubar));

    while ((converged == false) && (iter < 30)) {
      iter++;
      X = X - R / dRdX;
      alpha = eqps + sqrt23 * X;
      H = K * alpha;
      dH = K;
      R = smag - (2.0 * mubar * X + sqrt23 * (Y + H));
      dRdX = -2.0 * mubar * (1.0 + dH / (3.0 * mubar));
      res = std::abs(R);
      if ((res < 1.0e-11) || (res / Y < 1.0e-11) || (res / f < 1.0e-11)) {
        converged = true;
      }
    }
    OMEGA_H_CHECK(converged == true);

    // updates
    dgam = X;
    auto const N = (1.0 / smag) * s;
    auto const fpinc = dgam * N;
    s -= 2.0 * mubar * fpinc;
    eqps = alpha;
    auto const Fpinc = lgr::exp::exp(fpinc);
    Fp = Fpinc * Fp;
  }

  // elastic increment all plastic quantities remain the same
  // update proceeds in the same manner as below

  // compute stress
  auto const p = 0.5 * kappa * (J - 1.0 / J);
  sigma = s / J + p * I;

  // compute wave speed (same as neohookean for now)
  auto tangent_bulk_modulus = 0.5 * kappa * (J + 1.0 / J);
  auto plane_wave_modulus = tangent_bulk_modulus + (4.0 / 3.0) * mu;
  OMEGA_H_CHECK(plane_wave_modulus > 0.0);
  c = std::sqrt(plane_wave_modulus / rho);
  OMEGA_H_CHECK(c > 0.0);

}


template <class Elem>
struct AlbanyJ2 : public Model<Elem> {

  FieldIndex poissons_ratio;
  FieldIndex elastic_modulus;
  FieldIndex hardening_modulus;
  FieldIndex yield_strength;
  FieldIndex plastic_def_grad;
  FieldIndex equivalent_plastic_strain;

  AlbanyJ2(Simulation& sim_in, Omega_h::InputMap& pl)
      : Model<Elem>(sim_in, pl) {
    this->poissons_ratio = this->point_define(
        "nu", "Poissons ratio", 1, RemapType::PICK, pl, "");
    this->elastic_modulus = this->point_define(
        "E", "elastic modulus", 1, RemapType::PICK, pl, "");
    this->hardening_modulus = this->point_define(
        "K", "hardening modulus", 1, RemapType::PICK, pl, "");
    this->yield_strength = this->point_define(
        "Y", "yield strength", 1, RemapType::PICK, pl, "");
    this->plastic_def_grad = this->point_define(
        "Fp", "plastic deformation gradient", square(this->sim.dim()),
        RemapType::PICK, pl, "I");
    this->equivalent_plastic_strain = this->point_define(
        "eqps", "equivalent plastic strain", 1, RemapType::PICK, pl, "0.0");
  }

  std::uint64_t exec_stages() override final { return AT_MATERIAL_MODEL; }
  char const* name() override final { return "Albany J2"; }

  void at_material_model() override final {

    // local vars
    auto points_to_nu = this->points_get(this->poissons_ratio);
    auto points_to_E = this->points_get(this->elastic_modulus);
    auto points_to_K = this->points_get(this->hardening_modulus);
    auto points_to_Y = this->points_get(this->yield_strength);
    auto points_to_Fp = this->points_getset(this->plastic_def_grad);
    auto points_to_eqps = this->points_getset(this->equivalent_plastic_strain);

    // simulation vars
    auto points_to_rho = this->points_get(this->sim.density);
    auto points_to_F = this->points_get(this->sim.def_grad);
    auto points_to_F_init = this->points_get(this->sim.def_grad_init);
    auto points_to_stress = this->points_set(this->sim.stress);
    auto points_to_c = this->points_set(this->sim.wave_speed);

    auto functor = OMEGA_H_LAMBDA(int point) {

      // get the total deformation gradient in 3x3 form
      auto F_small = getfull<Elem>(points_to_F, point);
      auto F_init_small = getfull<Elem>(points_to_F_init, point);
      auto F_tot_small = F_small * F_init_small;
      auto F = identity_tensor<3>();
      for (int i = 0; i < Elem::dim; ++i)
      for (int j = 0; j < Elem::dim; ++j)
        F(i, j) = F_tot_small(i, j);

      // get the old plastic history variables
      auto eqps = points_to_eqps[point];
      auto Fp_small = getfull<Elem>(points_to_Fp, point);
      auto Fp  = identity_tensor<3>();
      for (int i = 0; i < Elem::dim; ++i)
      for (int j = 0; j < Elem::dim; ++j)
        Fp(i, j) = Fp_small(i, j);

      // get the material properties
      auto rho = points_to_rho[point];
      auto E = points_to_E[point];
      auto nu = points_to_nu[point];
      auto K = points_to_K[point];
      auto Y = points_to_Y[point];

      // values to update
      double c;
      Tensor<3> sigma;

      // update the stress, wave speed, and plastic history vars
      albany_J2_update(rho, E, nu, K, Y, F, Fp, eqps, sigma, c);

      // resize Fp to its small value
      Tensor<Elem::dim> Fp_new;
      for (int i = 0; i < Elem::dim; ++i)
      for (int j = 0; j < Elem::dim; ++j)
        Fp_new(i, j) = Fp(i, j);

      // set new values
      setstress(points_to_stress, point, sigma);
      setfull(points_to_Fp, point, Fp_new);
      points_to_eqps[point] = eqps;
      points_to_c[point] = c;

    };
    parallel_for(this->points(), std::move(functor));

  }

};

template <class Elem>
ModelBase* albany_J2_factory(
    Simulation& sim, std::string const&, Omega_h::InputMap& pl) {
  return new AlbanyJ2<Elem>(sim, pl);
}

#define LGR_EXPL_INST(Elem)                                                    \
  template ModelBase* albany_J2_factory<Elem>(                                 \
      Simulation&, std::string const&, Omega_h::InputMap&);
LGR_EXPL_INST_ELEMS
#undef LGR_EXPL_INST

}  // namespace lgr
