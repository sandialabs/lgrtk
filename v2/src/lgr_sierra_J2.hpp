#ifndef LGR_SIERRA_J2_HPP
#define LGR_SIERRA_J2_HPP

#include <lgr_exp.hpp>
#include <lgr_element_types.hpp>
#include <lgr_model.hpp>
#include <string>

namespace lgr {

namespace {

template <typename Tensor, typename Scalar>
OMEGA_H_INLINE Scalar
               effective_stress(Tensor const& s, Scalar const& sigy)
{
  auto const sb   = s / sigy;
  auto const seff = Omega_h::inner_product(sb, sb);
  return std::sqrt(1.5 * seff) * sigy;
}

}  // anonymous namespace

OMEGA_H_INLINE void
sierra_J2_update(
    double const     rho,
    double const     E,
    double const     nu,
    double const     K,
    double const     beta,
    double const     Y,
    Tensor<3> const& F,
    Tensor<3>&       Fp,
    double&          eqps,
    Tensor<3>&       sigma,
    double&          c)
{
  // compute material properties
  auto const I       = Omega_h::identity_matrix<3, 3>();
  auto const mu      = E / (2.0 * (1.0 + nu));
  auto const twomu   = 2.0 * mu;
  auto const threemu = 3.0 * mu;
  auto const kappa   = E / (3.0 * (1.0 - 2.0 * nu));
  auto const lambda  = twomu * nu / (1.0 - 2.0 * nu);

  // get def grad quantities
  auto const J     = determinant(F);
  auto const Fpinv = invert(Fp);
  auto       Fe    = F * Fpinv;

  //
  // Predict stress
  //

  // elastic log strain: 1/2 log(Ce)
  auto const Ce = transpose(Fe) * Fe;
  auto const Ee = 0.5 * Omega_h::log_spd(Ce);

  // M - Mandel stress in intermediate config
  // s - deviatoric Mandel stress
  auto const trEe     = trace(Ee);
  auto       M        = lambda * trEe * I + twomu * Ee;
  auto       s        = M - trace(M) * I / 3.0;
  auto       pressure = kappa * trEe;
  M                   = pressure * I + s;

  // Cauchy stress
  sigma = transpose(invert(Fe)) * M * transpose(Fe) / J;

  // compute wave speed as same as for small strains (conservative)
  auto const tangent_bulk_modulus = kappa;
  auto const plane_wave_modulus   = tangent_bulk_modulus + (4.0 / 3.0) * mu;
  OMEGA_H_CHECK(plane_wave_modulus > 0.0);
  c = std::sqrt(plane_wave_modulus / rho);
  OMEGA_H_CHECK(c > 0.0);

  // Voce hardening
  auto       sbar      = Y + K * (1.0 - std::exp(-beta * eqps));
  auto const seff_pred = effective_stress(s, Y);

  // check for yielding
  if (seff_pred <= sbar) return;

  double       merit_old = 1.0;
  double       merit_new = 1.0;
  double       dg_tol    = 1.0;
  int          iter      = 0;
  double       eqps_new  = 0.0;
  const double tolerance = 1.0e-10;
  double       dg        = 0.0;

  // line search parameters
  double const eta_ls  = 0.1;
  double const beta_ls = 1.e-05;

  int const max_ls_iter  = 128;
  int const max_rma_iter = 128;

  // begin return mapping algorithm
  while (dg_tol > tolerance) {
    ++iter;
    double dg0                       = dg;
    eqps_new                         = eqps + dg0;
    auto const hprime                = beta * K * std::exp(-beta * eqps_new);
    auto const numerator             = seff_pred - threemu * dg0 - sbar;
    auto const denominator           = threemu + hprime;
    merit_old                        = numerator * numerator;
    auto const ddg                   = numerator / denominator;
    double     alpha_ls              = 1.0;
    int        line_search_iteration = 0;

    // line search
    bool line_search = true;
    while (line_search == true) {
      ++line_search_iteration;
      dg = dg0 + alpha_ls * ddg;
      if (dg < 0.0) dg = 0.0;
      eqps_new            = eqps + dg;
      sbar                = Y + K * (1.0 - std::exp(-beta * eqps_new));
      auto const residual = seff_pred - sbar - threemu * dg;
      merit_new           = residual * residual;
      auto const factor   = 1.0 - 2.0 * beta_ls * alpha_ls;
      if (merit_new <= factor * merit_old) {
        merit_old   = merit_new;
        line_search = false;
      } else {
        auto const alpha_ls_old = alpha_ls;
        alpha_ls                = alpha_ls_old * alpha_ls_old * merit_old /
                   (merit_new - merit_old + 2.0 * alpha_ls_old * merit_old);
        if (eta_ls * alpha_ls_old > alpha_ls) {
          alpha_ls = eta_ls * alpha_ls_old;
        }
      }
      if (line_search_iteration > max_ls_iter && line_search == true) {
        Omega_h_fail("Line search failing in Sierra J2 model.\n");
      }
    }  // end line search
    dg_tol = std::sqrt(0.5 * merit_new / twomu / twomu);
    if (iter >= max_rma_iter) {
      Omega_h_fail("Return mapping not converging in Sierra J2 model.\n");
    }

  }  // end return mapping algorithm

  auto const n = 1.5 * s / seff_pred;
  auto const A = dg * n;
  s -= twomu * A;

  eqps = eqps_new;
  Fp   = lgr::exp::exp(A) * Fp;
  Fe   = F * invert(Fp);

  // update stress
  pressure = kappa * trace(Ee);
  M        = pressure * I + s;

  // Cauchy stress
  sigma = transpose(invert(Fe)) * M * transpose(Fe) / J;

  return;
}


template <class Elem>
ModelBase*
sierra_J2_factory(
    Simulation&        sim,
    std::string const& name,
    Omega_h::InputMap& pl);

#define LGR_EXPL_INST(Elem)                           \
  extern template ModelBase* sierra_J2_factory<Elem>( \
      Simulation&, std::string const&, Omega_h::InputMap&);
LGR_EXPL_INST_ELEMS
#undef LGR_EXPL_INST

}  // namespace lgr

#endif
