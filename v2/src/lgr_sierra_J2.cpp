#include <lgr_sierra_J2.hpp>
#include <lgr_exp.hpp>
#include <lgr_for.hpp>
#include <lgr_simulation.hpp>

namespace lgr {

namespace {

template<typename Tensor, typename Scalar>
OMEGA_H_INLINE Scalar effective_stress(Tensor const & s, Scalar const & sigy)
{
  auto const sb = s / sigy;
  Scalar const seff = Omega_h::inner_product(sb, sb);
  return std::sqrt(1.5 * seff) * sigy;
}


} // anonymous namespace

OMEGA_H_INLINE void sierra_J2_update(
    double const rho,
    double const E,
    double const nu,
    double const K,
    double const beta,
    double const Y,
    Tensor<3> const& F,
    Tensor<3>& Fp,
    double& eqps,
    Tensor<3>& sigma,
    double& c){

  // line search parameters
  double const eta_ls  = 0.1;
  double const beta_ls = 1.e-05;

  int const max_ls_iter  = 16;
  int const max_rma_iter = 16;

  auto const I = Omega_h::identity_matrix<3, 3>();

  // compute material properties
  auto const mu = E / (2.0 * (1.0 + nu));
  auto const twomu = 2.0 * mu;
  auto const threemu = 3.0 * mu;
  auto const kappa = E / (3.0 * (1.0 - 2.0 * nu));
  auto const lambda = twomu * nu / (1.0 - 2.0 * nu);

  // get def grad quantities
  auto const J = determinant(F);
  auto const Fpinv = invert(Fp);
  auto Fe = F * Fpinv;

  //
  // Predict stress
  //

  // elastic log strain: 1/2 log(Ce)
  auto const Ce = transpose(Fe) * Fe;
  auto const Ee = Omega_h::log_spd(Ce);

  // M - Mandel stress in intermediate config
  // s - deviatoric Mandel stress
  auto const trEe = trace(Ee);
  auto M = lambda * trEe * I + 2.0 * mu * Ee;
  auto s = M - trace(M) / 3.0 * I;
  auto pressure = kappa * trEe;
  M = pressure * I + s;

  // cauchy stress = (1/J)Fe^(-T)*M*Fe^T
  sigma = transpose(invert(Fe)) * M * transpose(Fe) / J;

  // compute wave speed (same as neohookean for now)
  auto const tangent_bulk_modulus = 0.5 * kappa * (J + 1.0 / J);
  auto const plane_wave_modulus = tangent_bulk_modulus + (4.0 / 3.0) * mu;
  OMEGA_H_CHECK(plane_wave_modulus > 0.0);
  c = std::sqrt(plane_wave_modulus / rho);
  OMEGA_H_CHECK(c > 0.0);

  // Voce hardening
  auto sbar = Y + K * (1.0 - std::exp(-beta * eqps));
  auto const seff_pred = effective_stress(s, Y);

  // check for yielding
  if (seff_pred <=  sbar) return;

  double merit_old = 1.0;
  double merit_new = 1.0;
  double dg_tol = 1.0;
  int iter = 0;
  double eqps_new = 0.0;
  const double tolerance = 1.0e-10;
  double dg = 0.0;

  // begin return mapping algorithm
  while (dg_tol > tolerance) {
    ++iter;
    double dg0 = dg;
    eqps_new = eqps + dg0;
    auto const hprime = beta * K * std::exp(-beta * eqps_new);
    auto const numerator = seff_pred - threemu * dg0 - sbar;
    auto const denominator = threemu + hprime;
    auto const ddg = numerator / denominator;
    double alpha_ls = 1.0;
    int line_search_iteration = 0;

    // Line search
    bool line_search = true;
    while (line_search == true) {
      ++line_search_iteration;
      dg = dg0 + alpha_ls * ddg;
      if (dg < 0.0) dg = 0.0;
      eqps_new = eqps + dg;
      sbar = Y + K * (1.0 - std::exp(-beta * eqps_new));
      auto const residual =  seff_pred - sbar - threemu * dg;
      merit_new = residual * residual;
      auto const factor = 1.0 - 2.0 * beta_ls * alpha_ls;
      if (merit_new <= factor*merit_old) {
        merit_old = merit_new;
        line_search = false;
      } else {
        auto const alpha_ls_old = alpha_ls;
        alpha_ls = alpha_ls_old * alpha_ls_old * merit_old /
            (merit_new - merit_old + 2.0 * alpha_ls_old * merit_old);
        if (eta_ls * alpha_ls_old > alpha_ls) {
          alpha_ls = eta_ls * alpha_ls_old;
        }
      }
      if (line_search_iteration > max_ls_iter && line_search == true) {
        Omega_h_fail("Line search in Sierra J2 model");
      }
    } // End line search
    dg_tol = std::sqrt(0.5 * merit_new / twomu / twomu);
    if (iter >= max_rma_iter) {
      Omega_h_fail("Return mapping algorithm in Sierra J2 model");
    }

  } // End return mapping algorithm

  auto const n = 1.5 * s / seff_pred;
  auto const A = dg * n;
  s -= twomu * A;

  eqps = eqps_new;
  Fp = lgr::exp::exp(A) * Fp;
  Fe = F * invert(Fp);

  // udpate stress
  pressure = kappa * trace(Ee);
  M = s + pressure*I;

  // cauchy stress = (1/J)Fe^(-T)*M*Fe^T
  sigma = transpose(invert(Fe)) * M * transpose(Fe) / J;

  return;
}


template <class Elem>
struct SierraJ2 : public Model<Elem> {

  FieldIndex poissons_ratio;
  FieldIndex elastic_modulus;
  FieldIndex hardening_modulus;
  FieldIndex hardening_exponent;
  FieldIndex yield_strength;
  FieldIndex plastic_def_grad;
  FieldIndex equivalent_plastic_strain;

  SierraJ2(Simulation& sim_in, Omega_h::InputMap& pl)
      : Model<Elem>(sim_in, pl) {
    this->poissons_ratio = this->point_define(
        "nu", "Poissons ratio", 1, RemapType::PICK, pl, "");
    this->elastic_modulus = this->point_define(
        "E", "elastic modulus", 1, RemapType::PICK, pl, "");
    this->hardening_modulus = this->point_define(
        "K", "hardening modulus", 1, RemapType::PICK, pl, "");
    this->hardening_exponent = this->point_define(
        "beta", "hardening exponent", 1, RemapType::PICK, pl, "");
    this->yield_strength = this->point_define(
        "Y", "yield strength", 1, RemapType::PICK, pl, "");
    this->plastic_def_grad = this->point_define(
        "Fp", "plastic deformation gradient", square(this->sim.dim()),
        RemapType::PICK, pl, "I");
    this->equivalent_plastic_strain = this->point_define(
        "eqps", "equivalent plastic strain", 1, RemapType::PICK, pl, "0.0");
  }

  std::uint64_t exec_stages() override final { return AT_MATERIAL_MODEL; }
  char const* name() override final { return "Sierra J2"; }

  void at_material_model() override final {

    // local vars
    auto points_to_nu = this->points_get(this->poissons_ratio);
    auto points_to_E = this->points_get(this->elastic_modulus);
    auto points_to_K = this->points_get(this->hardening_modulus);
    auto points_to_beta = this->points_get(this->hardening_exponent);
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
      auto beta = points_to_beta[point];
      auto Y = points_to_Y[point];

      // values to update
      double c;
      Tensor<3> sigma;

      // update the stress, wave speed, and plastic history vars
      sierra_J2_update(rho, E, nu, K, beta, Y, F, Fp, eqps, sigma, c);

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
ModelBase* sierra_J2_factory(
    Simulation& sim, std::string const&, Omega_h::InputMap& pl) {
  return new SierraJ2<Elem>(sim, pl);
}

#define LGR_EXPL_INST(Elem)                                                    \
  template ModelBase* sierra_J2_factory<Elem>(                                 \
      Simulation&, std::string const&, Omega_h::InputMap&);
LGR_EXPL_INST_ELEMS
#undef LGR_EXPL_INST

}  // namespace lgr
