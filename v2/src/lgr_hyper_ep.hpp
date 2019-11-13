#ifndef LGR_HYPER_EP_HPP
#define LGR_HYPER_EP_HPP

#include <string>
#include <sstream>

#include <lgr_element_types.hpp>
#include <lgr_mie_gruneisen.hpp>
#ifdef LGR_HAVE_SIERRA_J2
#include <lgr_sierra_J2.hpp>
#endif
#include <lgr_model.hpp>

namespace lgr {

namespace hyper_ep {


OMEGA_H_INLINE double
absmax(Tensor<3> a) {
  double m = std::abs(a(0,0));
  for (int i=1; i<3; i++)
    m = Omega_h::max2(m, std::abs(a(i,i)));
  return m;
}


OMEGA_H_INLINE bool
is_deviatoric(Tensor<3> const a){
  return std::abs(trace(a)) <= 1.0e-12;
}


OMEGA_H_INLINE Tensor<3>
deviatoric_part(Tensor<3> const a)
{
  auto tr = trace(a) / 3.0;
  auto const I = identity_matrix<3, 3>();
  auto dev = a - tr / 3.0 * I;
  // Manage round-off from working with large numbers by "re-deviating" the
  // deviator.
  tr = trace(dev);
  if (std::abs(tr) > 1.0e-14) {
    dev(0,0) = dev(0,0) - tr / 3.0;
    dev(1,1) = dev(1,1) - tr / 3.0;
    dev(2,2) = -(dev(0,0) + dev(1,1));
  }
  if (!is_deviatoric(dev))
  {
    printf(
        "The following matrix was not made deviatoric!:\n"
        "\tbefore => [%g, %g, %g, %g, %g, %g], trace = %g\n"
        "\t after => [%g, %g, %g, %g, %g, %g], trace = %g\n",
        a(0,0), a(1,1), a(2,2), a(0,1), a(1,2), a(0,2), trace(a),
        dev(0,0), dev(1,1), dev(2,2), dev(0,1), dev(1,2), dev(0,2), trace(dev)
    );
    OMEGA_H_CHECK(false);
  }
  return dev;
}


enum class Elastic
{
  MANDEL
};

enum class Hardening
{
  NONE,
  LINEAR_ISOTROPIC,
  POWER_LAW,
  ZERILLI_ARMSTRONG,
  JOHNSON_COOK,
#ifdef LGR_HAVE_SIERRA_J2
  SIERRA_J2
#endif
};

enum class RateDependence
{
  NONE,
  ZERILLI_ARMSTRONG,
  JOHNSON_COOK
};

enum class Damage
{
  NONE,
  JOHNSON_COOK
};

enum class EOS {
  NONE,
  MIE_GRUNEISEN
};

struct Properties {
  // Elasticity
  Elastic elastic;
  double E;
  double Nu;

  // Plasticity
  Hardening hardening;
  RateDependence rate_dep;
  double p0;
  double p1;  // Hardening modulus
  double p2;  // exponent in hardening
  double p3;
  double p4;
  double p5;
  double p6;
  double p7;
  double p8;

  // Damage parameters
  Damage damage;
  bool allow_no_tension;
  bool allow_no_shear;
  bool set_stress_to_zero;
  double D1;
  double D2;
  double D3;
  double D4;
  double D5;
  double D6;
  double D7;
  double D8;
  double D0;
  double DC;

  // Equation of state
  EOS eos;
  double rho0;
  double gamma0;
  double cs;
  double s1;
  double e0;

  Properties()
      : elastic(Elastic::MANDEL),
        hardening(Hardening::NONE),
        rate_dep(RateDependence::NONE),
        damage(Damage::NONE),
        allow_no_tension(true),
        allow_no_shear(false),
        set_stress_to_zero(false),
        eos(EOS::NONE) {}
};

using tensor_type = Matrix<3, 3>;

void read_and_validate_elastic_params(
    Omega_h::InputMap& params, Properties& props);
void read_and_validate_plastic_params(
    Omega_h::InputMap& params, Properties& props);
void read_and_validate_damage_params(
    Omega_h::InputMap& params, Properties& props);
void read_and_validate_eos_params(
    Omega_h::InputMap& params, Properties& props);


/** \brief Constant yield strength
 *
 * Constant J2 plasticity with yield strength given by
 *
 *     Y = Y0
 *
 * where Y0 is a material parameter (yield in uniaxial tension)
*/
OMEGA_H_INLINE void
hard_constant(double& Y, double& dY,
    double const /* ep */, double const /* epdot */, double const /* temp */,
    double const Y0)
{
  Y = Y0;
  dY = 0.0;
}

/** \brief Linear isotropic hardening.
 *
 * Yield strength given by
 *
 *     Y = Y0 + H ep
 *
 * where Y0 and H are material parameters (yield in uniaxial tension and
 * constant hardening modulus, respecitvely) and ep is the current equivalent
 * plastic strain.
 */
OMEGA_H_INLINE void
hard_linear_isotropic(double& Y, double& dY,
    double const ep, double const /* epdot */, double const /* temp */,
    double const A, double const B)
{
  Y = A + B * ep;
  dY = B;
}

OMEGA_H_INLINE void
hard_power_law(double& Y, double& dY,
    double const ep, double const /* epdot */, double const /* temp */,
    double const A, double const B, double const n)
{
  Y = (ep > 0.0) ? (A + B * std::pow(ep, n)) : A;
  dY = (ep > 0.0) ? n * B * std::pow(ep, (n - 1)) : 0.0;
}

/** \brief Johnson-Cook yield strength
 *
 * Implements the Johnson-Cook strength model, namely
 *
 *     Y = (A + B ep^n) (1 + C log(R)) (1 - th^m)
 *
 * where A, B, C, n, and m are material properties and the terms R and th are
 * given by
 *
 *     R = epdot / epdot0
 *     th = (T - T0) / (TM - T0)
 *
 * and epdot0, T0, and TM are material properties
*/
OMEGA_H_INLINE void
hard_johnson_cook(double& Y, double& dY,
    double const ep, double const epdot, double const temp,
    double const A, double const B, double const n,
    double const T0, double const TM, double const m,
    double const C, double const epdot0)
{

  // Constant and plastic strain contribution
  double f = A + B * std::pow(ep, n);
  double df = (ep > 0.0) ? (B * n * std::pow(ep, n - 1)) : 0.0;

  // Temperature contribution
  double th = 0.0;
  if (temp >= T0 && temp <= TM)
    th = (temp - T0) / (TM - T0);
  else if (temp > TM)
    th = 1.0;
  double g = (th > 1.e-14) ? 1.0 - std::pow(th, m) : 1.0;
  double dg = 0.0;

  // Rate of plastic strain contribution
  double h = 1.0;
  double dh = 0.0;
  if ((epdot0 > 0.0) && (C > 0.0)) {
    auto const rfac = epdot / epdot0;
    if (rfac < 1.0) {
      h = std::pow((1.0 + rfac), C);
    } else {
      h = 1.0 + C * std::log(rfac);
    }
  }
  Y = f * g * h;
  dY = (Y - A) * (df / f + dg / g + dh / h);
}


/** \brief Johnson-Cook yield strength
 *
 * Implements a modified Zerilli-Armstrong model, namely
 *
 *     Y = (A + B ep^n) + C sqrt(ep) e^(-alpha T) + D e^(-beta T)
 *
 * with
 *
 *     alpha = alpha_0 - alpha_1 log(epdot)
 *     beta = beta_0 - beta_1 log(epdot)
 *
*/
OMEGA_H_INLINE void
hard_zerilli_armstrong(double& Y, double& dY,
    double const ep, double const epdot, double const temp,
    double const A, double const B, double const n,
    double const C, double const alpha_0, double const alpha_1,
    double const D, double const beta_0, double const beta_1)
{
  double const f = A + B * std::pow(ep, n);
  double const df = (std::abs(ep) < 1.0e-14) ? 0.0
                  : n * B * std::pow(ep, n - 1.0);

  double const alpha = alpha_0 - alpha_1 * std::log(epdot);
  double const g = C * std::sqrt(ep) * std::exp(-alpha * temp);
  double const dg = (std::abs(ep) < 1.0e-14) ? 0.0
                  : 0.5 * C * std::exp(-alpha * temp) / std::sqrt(ep);

  double const beta = beta_0 - beta_1 * std::log(epdot);
  double const h = D * std::exp(-beta * temp);
  double const dh = 0.0;

  Y = f + g + h;
  dY = df + dg + dh;
}

OMEGA_H_INLINE void
hard(double&Y, double& dY,
    double const ep, double const epdot, double const temp,
    Properties const props)
{
  if(props.hardening == Hardening::NONE) {
    hard_constant(Y, dY, ep, epdot, temp, props.p0);
  } else if (props.hardening == Hardening::LINEAR_ISOTROPIC) {
    hard_linear_isotropic(Y, dY, ep, epdot, temp, props.p0, props.p1);
  } else if (props.hardening == Hardening::POWER_LAW) {
    hard_power_law(Y, dY, ep, epdot, temp, props.p0, props.p1, props.p2);
  } else if (props.hardening == Hardening::ZERILLI_ARMSTRONG) {
    hard_zerilli_armstrong(Y, dY, ep, epdot, temp,
        props.p0, props.p1, props.p2, props.p3, props.p4,
        props.p5, props.p6, props.p7, props.p8);
  } else if (props.hardening == Hardening::JOHNSON_COOK) {
    hard_johnson_cook(Y, dY, ep, epdot, temp,
        props.p0, props.p1, props.p2, props.p3,
        props.p4, props.p5, props.p6, props.p7);
  }
}


OMEGA_H_INLINE void
scalar_damage_johnson_cook(double& dp, double const pres, Tensor<3> const s,
    double const /* ep */, double const epdot, double const temp, double const dtime,
    double const D1, double const D2, double const D3, double const D4,
    double const D5, double const D6, double const D7, double const eps_f_min)
{
  double tolerance = 1e-10;
  auto const smag = norm(s);
  auto const seq = std::sqrt(smag * smag * 1.5);

  double eps_f = eps_f_min;
  double sig_star = (std::abs(seq) > 1e-16) ? -pres / seq : 0.0;
  if (sig_star < 1.5) {
    // NOT SPALL
    // sig_star < 1.5 indicates spall conditions are *not* met and the failure
    // strain must be calculated.
    sig_star = Omega_h::max2(Omega_h::min2(sig_star, 1.5), -1.5);

    // Stress contribution to damage
    double stress_contrib = D1 + D2 * std::exp(D3 * sig_star);

    // Strain rate contribution to damage
    double dep_contrib = 1.0;
    if (epdot < 1.0) {
      dep_contrib = std::pow((1.0 + epdot), D4);
    } else {
      dep_contrib = 1.0 + D4 * std::log(epdot);
    }

    double temp_contrib = 1.0;
    auto const temp_ref = D6;
    auto const temp_melt = D7;
    if (std::abs(temp_melt-Omega_h::ArithTraits<double>::max())+1.0 != 1.0) {
      auto const tstar =
          temp > temp_melt ? 1.0 : (temp - temp_ref) / (temp_melt - temp_ref);
      temp_contrib += D5 * tstar;
    }

    // Calculate the updated scalar damage parameter
    eps_f = stress_contrib * dep_contrib * temp_contrib;
  }

  if (eps_f < tolerance) return;

  // Calculate plastic strain increment
  auto const dep = epdot * dtime;
  auto const ddp = dep / eps_f;
  dp = (dp + ddp < tolerance) ? 0.0 : dp + ddp;
}


OMEGA_H_INLINE void
scalar_damage(double& dp, double const pres, Tensor<3> const s,
    double const ep, double const epdot, double const temp, double const dtime,
    Properties const props)
{
  if (props.damage == Damage::JOHNSON_COOK) {
    scalar_damage_johnson_cook(dp, pres, s, ep, epdot, temp, dtime,
        props.D1, props.D2, props.D3, props.D4, props.D5, props.D6, props.D7, props.D8);
  }
}


// Returns the Mandel stress
OMEGA_H_INLINE void
mandel(double& pres, Tensor<3>& s,
    Tensor<3> const F, Tensor<3> const Fp, Properties const props)
{
  auto const Fe = F * invert(Fp);
  auto const kappa = props.E / (3.0 * (1.0 - 2.0 * props.Nu));
  auto const mu = props.E / 2.0 / (1.0 + props.Nu);
  auto const lambda = 2.0 * mu * props.Nu / (1.0 - 2.0 * props.Nu);

  // elastic log strain: 1/2 log(Ce)
  auto const Ce = transpose(Fe) * Fe;
  auto const Ee = 0.5 * Omega_h::log_spd(Ce);

  // M - Mandel stress in intermediate config
  // s - deviatoric Mandel stress
  auto const ev = trace(Ee);
  auto const I = Omega_h::identity_matrix<3, 3>();
  auto M = lambda * ev * I + 2.0 * mu * Ee;
  s = deviatoric_part(M);
  pres = -kappa * ev;
}


/** \brief High-level wrapper around elastic update */
OMEGA_H_INLINE void
elastic(double& pres, Tensor<3>& s, double& wave_speed,
    double const J, Tensor<3> const F, Tensor<3> const Fp,
    double const rho, Properties const props)
{
  if (props.elastic == Elastic::MANDEL)
  {
    mandel(pres, s, F, Fp, props);
  }
  else
  {
    Omega_h_fail("Unsupported elastic type");
  }
  OMEGA_H_CHECK(is_deviatoric(s));

  double K = 0.0;
  if (props.eos == EOS::MIE_GRUNEISEN) {
    // Replace pressure with that computed from EOS if needed.
    double c = 0.0;
    mie_gruneisen_update(props.rho0, props.gamma0, props.cs, props.s1,
                         rho, props.e0, pres, c);
    K = rho * c * c;
  } else {
    // wave speed
    auto kappa = props.E / (3.0 * (1.0 - 2.0 * props.Nu));
    K = 0.5 * kappa * (J + 1.0 / J);
  }
  auto const H = 3.0 * K * (1.0 - props.Nu) / (1 + props.Nu);
  OMEGA_H_CHECK(H > 0.0);
  wave_speed = std::sqrt(H / rho);
  OMEGA_H_CHECK(wave_speed > 0.0);
}


OMEGA_H_INLINE bool
at_yield(Tensor<3> const s,
    double const ep, double const epdot, double const temp,
    Properties const props)
{
  // check the yield condition
  auto const smag = Omega_h::norm(s);
  auto Y = Omega_h::ArithTraits<double>::max();
  double dY = 0.0;
  hard(Y, dY, ep, epdot, temp, props);
  double const sq23 = std::sqrt(2.0 / 3.0);
  auto const f = smag - sq23 * Y;
  return f > 1.0e-12;
}


/* Computes the radial return
 *
 * Yield function:
 *   S:S - Sqrt[2/3] * Y = 0
 * where S is the stress deviator.
 *
 * Equivalent plastic strain:
 *   ep = Integrate[Sqrt[2/3]*Sqrt[epdot:epdot], 0, t]
 *
 */
OMEGA_H_INLINE void
radial_return(Tensor<3>& s, double const /*J*/, Tensor<3> const /*F*/, Tensor<3>& Fp,
    double& ep, double& epdot, double const temp, double const /*dtime*/,
    Properties const props)
{
  OMEGA_H_CHECK(is_deviatoric(s));
  double const sq32 = std::sqrt(3.0 / 2.0);
  double const tol = 1.0E-10;

  auto const mu = props.E / (2.0 * (1.0 + props.Nu));
  auto const seff = sq32 * Omega_h::norm(s);

  auto Y = Omega_h::ArithTraits<double>::max();
  double dY = 0.0;
  hard(Y, dY, ep, epdot, temp, props);
  if (seff <= Y)
    return;

  double dep = 0.0;
  double alpha  = 0.0;

  // line search parameters
  double const eta_ls = 0.1;
  double const beta_ls = 1.e-05;

  bool conv1 = false;
  int const max_rma_iter = 128;
  for (int iter=0; iter<max_rma_iter; iter++)
  {

    alpha = ep + dep;

    auto const g = seff - (3.0 * mu * dep + Y);
    auto const dg = -3.0 * mu - dY;
    auto const dgamma = -g / dg;

    // line search
    auto merit_old = g * g;
    auto merit_new = 1.0;
    auto dep0 = dep;
    auto alpha_ls = 1.0;

    bool conv2 = false;
    int const max_ls_iter = 128;
    for (int ls_iter=0; ls_iter<max_ls_iter; ls_iter++)
    {

      dep = dep0 + alpha_ls * dgamma;
      if (dep < 0.0) dep = 0.0;

      alpha = ep + dep;
      hard(Y, dY, alpha, epdot, temp, props);

      auto const R = seff - Y - 3.0 * mu * dep;
      merit_new = R * R;
      auto const factor = 1.0 - 2.0 * beta_ls * alpha_ls;
      if (merit_new <= factor * merit_old)
      {
        conv2 = true;
        break;
      }

      auto const alpha_ls_old = alpha_ls;
      alpha_ls = alpha_ls_old * alpha_ls_old * merit_old /
                 (merit_new - merit_old + 2.0 * alpha_ls_old * merit_old);
      if (eta_ls * alpha_ls_old > alpha_ls) {
        alpha_ls = eta_ls * alpha_ls_old;
      }
    }  // end line search
    if (!conv2)
      Omega_h_fail("Line search failed in Hyper EP model.\n");

    auto const dep_tol = std::sqrt(0.5 * merit_new / (2.0 * mu) / (2.0 * mu));
    if (dep_tol <= tol)
    {
      conv1 = true;
      break;
    }
  }

  if (!conv1)
    Omega_h_fail("Radial return not converged in Hyper EP model.\n");

  // updates
  auto const N = 1.5 * s / seff;
  auto const A = dep * N;
  s -= 2.0 * mu * A;
  ep = alpha;
  Fp = Omega_h::exp_spd(A) * Fp;
}


OMEGA_H_INLINE void
update_damage(double& pres, Tensor<3>& s, double& dp, int& localized,
    double const ep, double const epdot, double const temp, double const dtime,
    Properties const props)
{

  assert(false); // this has not yet been tested!

  auto const I = identity_matrix<3, 3>();
  auto T = s - pres * I;

  // Update damage
  scalar_damage(dp, pres, s, ep, epdot, temp, dtime, props);

  bool is_localized = false;
  if (props.damage != Damage::NONE) {
    // If the particle has already failed, apply various erosion algorithms
    if (localized > 0.0) {
      if(props.allow_no_tension) {
        if (pres < 0.0) {
          T = 0.0 * I;
        } else {
          T = -pres * I;
        }
      } else if(props.allow_no_shear) {
        T = -pres * I;
      } else if(props.set_stress_to_zero) {
        T = 0.0 * I;
      }
    }

    // Update damage and check modified TEPLA rule
    scalar_damage(dp, pres, s, ep, epdot, temp, dtime, props);

    double const por = 0.0;
    double const por_crit = 1.0;
    double const por_ratio = por / por_crit;
    double const D0dpDC = (props.D0 + dp) / props.DC;
    auto const tepla = por_ratio * por_ratio + D0dpDC * D0dpDC;
    if (tepla > 1.0) {
      is_localized = true;
    }
  }

  if (is_localized) {
    // If the localized material point fails again, set the stress to zero
    if (localized > 0.0) {
      dp = 0.0;
      T = 0.0 * I;
    } else {
      // set the particle localization flag to true
      localized = 1.0;
      dp = 0.0;
      // Apply various erosion algorithms
      if(props.allow_no_tension) {
        if (pres < 0.0) {
          T = 0.0 * I;
        } else {
          T = -pres * I;
        }
      } else if(props.allow_no_shear) {
        T = -pres * I;
      } else if(props.set_stress_to_zero) {
        T = 0.0 * I;
      }
    }
  }
}


OMEGA_H_INLINE_BIG void
update(Properties const props, double const rho, Tensor<3> const F,
    double const dtime, double const temp, Tensor<3>& T, double& wave_speed,
    Tensor<3>& Fp, double& ep, double& epdot, double& /*dp*/, double& /*localized*/)
{

#ifdef LGR_HAVE_SIERRA_J2
  if (props.hardening == Hardening::SIERRA_J2)
  {
    sierra_J2_update(rho, props.E, props.Nu, props.p1, props.p2, props.p0,
      F, Fp, ep, T, wave_speed);
    if (props.eos == EOS::MIE_GRUNEISEN)
    {
      double pres = 0.0;
      double c = 0.0;
      mie_gruneisen_update(props.rho0, props.gamma0, props.cs, props.s1,
                           rho, props.e0, pres, c);
      for (int i=0; i<3; i++) T(i,i) = -pres;
      auto const H = 3.0 * (rho * c * c) * (1.0 - props.Nu) / (1 + props.Nu);
      OMEGA_H_CHECK(H > 0.0);
      wave_speed = std::sqrt(H / rho);
      OMEGA_H_CHECK(wave_speed > 0.0);
    }
    return;
  }
#endif

  // compute material properties
  auto const J = determinant(F);

  Tensor<3> s;
  double pres = 0.0;

  // Determine the stress predictor.
  elastic(pres, s, wave_speed, J, F, Fp, rho, props);

  // check the yield condition
  if (at_yield(s, ep, epdot, temp, props))
    radial_return(s, J, F, Fp, ep, epdot, temp, dtime, props);

  // compute stress
  auto const I = Omega_h::identity_matrix<3, 3>();
  auto const M = s - pres * I;
  auto const Fe = F * invert(Fp);
  T = transpose(invert(Fe)) * M * transpose(Fe) / J;
}

}  // namespace hyper_ep

void setup_hyper_ep(Simulation& sim, Omega_h::InputMap& pl);

}  // namespace lgr

#endif  // LGR_HYPER_EP_HPP
