#ifndef LGR_HYPER_EP_HPP
#define LGR_HYPER_EP_HPP

#include <string>
#include <sstream>

#include <lgr_element_types.hpp>
#include <lgr_model.hpp>

namespace lgr {

namespace HyperEPDetails {

enum class ErrorCode {
  NOT_SET,
  SUCCESS,
  LINEAR_ELASTIC_FAILURE,
  HYPERELASTIC_FAILURE,
  RADIAL_RETURN_FAILURE,
  ELASTIC_DEFORMATION_UPDATE_FAILURE,
  MODEL_EVAL_FAILURE
};

enum class Elastic {
  LINEAR_ELASTIC,
  NEO_HOOKEAN
};

enum class Hardening {
  NONE,
  LINEAR_ISOTROPIC,
  POWER_LAW,
  ZERILLI_ARMSTRONG,
  JOHNSON_COOK
};

enum class RateDependence {
  NONE,
  ZERILLI_ARMSTRONG,
  JOHNSON_COOK
};

enum class StateFlag {
  NONE,
  TRIAL,
  ELASTIC,
  PLASTIC,
  REMAPPED
};

using tensor_type = Matrix<3, 3>;

char const* get_error_code_string(ErrorCode code);
void read_and_validate_elastic_params(
    Teuchos::ParameterList& params,
    std::vector<double>& props,
    Elastic& elastic);
void read_and_validate_plastic_params(
    Teuchos::ParameterList& params,
    std::vector<double>& props,
    Hardening& hardening,
    RateDependence& rate_dep);

/** \brief Determine the square of the left stretch B=V.V

Parameters
----------
tau : ndarray
    The Kirchhoff stress
mu : float
    The shear modulus

Notes
-----
On unloading from the current configuration, the left stretch V is recovered.
For materials with an isotropic fourth order elastic stiffness, the square of
the stretch is related to the Kirchhoff stress by

                       dev(tau) = mu dev(BB)                 (1)

where BB is J**(-2/3) B. Since det(BB) = 1 (1) can then be solved for BB
uniquely.

This routine solves the following nonlinear problem with local Newton
iterations

                      Solve:       Y = dev(X)
                      Subject to:  det(X) = 1

where Y = dev(tau) / mu
*/

OMEGA_H_INLINE
tensor_type find_bbe(tensor_type const tau, double const mu) {
  constexpr int maxit = 25;
  constexpr double tol = 1e-12;
  auto const txx = tau(0,0);
  auto const tyy = tau(1,1);
  auto const tzz = tau(2,2);
  auto const txy = .5 * (tau(0,1) + tau(1,0));
  auto const txz = .5 * (tau(0,2) + tau(2,0));
  auto const tyz = .5 * (tau(1,2) + tau(2,1));
  auto Be = Omega_h::deviator(tau) / mu;
  double bzz_old = 1;
  double bzz_new = 1;
  for (int i=0; i<maxit; i++) {
    // computes det(BBe), where BBe is the iscohoric deformation
    auto const fun_val =
      (bzz_old * mu * (-txy * txy + (bzz_old*mu + txx - tzz)*(bzz_old*mu + tyy - tzz))
      + 2*txy*txz*tyz + txz*txz*(-bzz_old*mu - tyy + tzz)
      + tyz*tyz*(-bzz_old*mu - txx + tzz)) / (mu*mu*mu);
    // computes d(det(BBe) - 1)/d(be_zz), where BBe is the iscohoric deformation
    auto const dfun_val =
      (bzz_old*mu*(2.0*bzz_old*mu + txx + tyy - 2.0*tzz)
      - txy*txy - txz*txz - tyz*tyz
      + (bzz_old*mu + txx - tzz)*(bzz_old*mu + tyy - tzz))/(mu*mu);
    bzz_new = bzz_old - (fun_val - 1.0) / dfun_val;
    Be(0,0) = (1.0 / mu) * (mu * bzz_new + txx - tzz);
    Be(1,1) = (1.0 / mu) * (mu * bzz_new + tyy - tzz);
    Be(2,2) = bzz_new;
    if (square(bzz_new - bzz_old) < tol) {
      return Be;
    }
    bzz_old = bzz_new;
  }
  OMEGA_H_NORETURN(tensor_type());
}

OMEGA_H_INLINE
double flow_stress(
    Hardening const hardening,
    RateDependence const rate_dep,
    std::vector<double> const& props,
    double const temp,
    double const ep,
    double const epdot) {
  auto Y = Omega_h::ArithTraits<double>::max();
  if (hardening == Hardening::NONE) {
    Y = props[2];
  } else if (hardening == Hardening::LINEAR_ISOTROPIC) {
    Y = props[2] + props[3] * ep;
  } else if (hardening == Hardening::POWER_LAW) {
    auto const a = props[2];
    auto const b = props[3];
    auto const n = props[4];
    Y = (ep > 0.0) ? (a + b * std::pow(ep, n)) : a;
  } else if (hardening == Hardening::ZERILLI_ARMSTRONG) {
    auto const a = props[2];
    auto const b = props[3];
    auto const n = props[4];
    Y = (ep > 0.0) ? (a + b * std::pow(ep, n)) : a;
    auto const c1 = props[5];
    auto const c2 = props[6];
    auto const c3 = props[7];
    auto alpha = c3;
    if (rate_dep == RateDependence::ZERILLI_ARMSTRONG) {
      auto const c4 = props[8];
      alpha -= c4 * std::log(epdot);
    }
    Y += (c1 + c2 * std::sqrt(ep)) * std::exp(-alpha * temp);
  } else if (hardening == Hardening::JOHNSON_COOK) {
    auto const ajo = props[2];
    auto const bjo = props[3];
    auto const njo = props[4];
    auto const temp_ref = props[5];
    auto const temp_melt = props[6];
    auto const mjo = props[7];
    // Constant contribution
    Y = ajo;
    // Plastic strain contribution
    if (bjo > 0.0) {
      Y += (fabs(njo) > 0.0) ? bjo * std::pow(ep, njo) : bjo;
    }
    // Temperature contribution
    if (fabs(temp_melt - std::numeric_limits<double>::max()) + 1.0 != 1.0) {
      double tstar = (temp > temp_melt) ? 1.0 : ((temp - temp_ref) / (temp_melt - temp_ref));
      Y *= (tstar < 0.0) ? (1.0 - tstar) : (1.0 - std::pow(tstar, mjo));
    }
  }
  if (rate_dep == RateDependence::JOHNSON_COOK) {
    auto const cjo = props[8];
    auto const epdot0 = props[9];
    auto const rfac = epdot / epdot0;
    // FIXME: This assumes that all the
    // strain rate is plastic.  Should
    // use actual strain rate.
    // Rate of plastic strain contribution
    if (cjo > 0.0) {
      Y *= (rfac < 1.0) ?  std::pow((1.0 + rfac), cjo) : (1.0 + cjo * std::log(rfac));
    }
  }
  return Y;
}

OMEGA_H_INLINE
double
dflow_stress(const Hardening& hardening,
             const RateDependence& rate_dep,
             const std::vector<double>& props,
             const double& temp, const double& ep,
             const double& epdot, const double& dtime)
{
  double deriv = 0.;
  if (hardening == Hardening::LINEAR_ISOTROPIC) {
    double b = props[3];
    deriv = b;
  }

  else if (hardening == Hardening::POWER_LAW) {
    double b = props[3];
    double n = props[4];
    deriv = (ep > 0.0) ? b * n * std::pow(ep, n - 1) : 0.0;
  }

  else if (hardening == Hardening::ZERILLI_ARMSTRONG) {
    double b = props[3];
    double n = props[4];
    deriv = (ep > 0.0) ? b * n * std::pow(ep, n - 1) : 0.0;

    double c1 = props[5];
    double c2 = props[6];
    double c3 = props[7];

    double alpha = c3;
    if (rate_dep == RateDependence::ZERILLI_ARMSTRONG) {
      double c4 = props[8];
      alpha -= c4 * std::log(epdot);
    }
    deriv += .5 * c2 / std::sqrt(ep <= 0.0 ? 1.e-8 : ep) * std::exp(-alpha * temp);

    if (rate_dep == RateDependence::ZERILLI_ARMSTRONG) {
      double c4 = props[8];
      double term1 = c1 * c4 * temp * std::exp(-alpha * temp);
      double term2 = c2 * sqrt(ep) * c4 * temp * std::exp(-alpha * temp);
      deriv += (term1 + term2) / (epdot <= 0.0 ? 1.e-8 : epdot) / dtime;
    }
  }

  else if (hardening == Hardening::JOHNSON_COOK) {
    double bjo = props[3];
    double njo = props[4];
    double temp_ref = props[5];
    double temp_melt = props[6];
    double mjo = props[7];

    // Calculate temperature contribution
    double temp_contrib = 1.0;
    if (fabs(temp_melt - std::numeric_limits<double>::max()) + 1.0 != 1.0) {
      double tstar = (temp > temp_melt) ? 1.0 : (temp - temp_ref) / (temp_melt - temp_ref);
      temp_contrib = (tstar < 0.0) ? (1.0 - tstar) : (1.0 - pow(tstar, mjo));
    }

    deriv = (ep > 0.0) ? (bjo * njo * pow(ep, njo - 1) * temp_contrib) : 0.0;

    if (rate_dep == RateDependence::JOHNSON_COOK) {
      double ajo = props[2];
      double cjo = props[8];
      double epdot0 = props[9];
      double rfac = epdot / epdot0;

      // Calculate strain rate contribution
      double term1 = (rfac < 1.0) ?  (std::pow((1.0 + rfac), cjo)) : (1.0 + cjo * std::log(rfac));

      double term2 = (ajo + bjo * std::pow(ep, njo)) * temp_contrib;
      if (rfac < 1.0) {
        term2 *= cjo * pow((1.0 + rfac), (cjo - 1.0));
      }
      else {
        term2 *= cjo / rfac;
      }
      deriv *= term1;
      deriv += term2 / dtime;
    }
  }
  constexpr double sq23 = 0.8164965809277261;
  return sq23 * deriv;
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
OMEGA_H_INLINE
ErrorCode
radial_return(const Hardening& hardening,
              const RateDependence& rate_dep,
              const std::vector<double>& props,
              const tensor_type& Te, const tensor_type& F,
              const double& temp, const double& dtime,
              tensor_type& T, tensor_type& Fp, double& ep, double& epdot,
              StateFlag& flag)
{
  const double tol1 = 1e-12;
  const double tol2 = std::min(dtime, 1e-6);
  constexpr double twothird = 2.0 / 3.0;
  const double sq2 = sqrt(2.0);
  const double sq3 = sqrt(3.0);
  const double sq23 = sq2 / sq3;
  const double sq32 = 1.0 / sq23;

  const double E = props[0];
  const double Nu = props[1];
  const double mu = E / 2.0 / (1. + Nu);
  const double twomu = 2. * mu;
  double gamma = epdot * dtime * sq32;

  // Possible states at this point are TRIAL or REMAPPED
  if (flag != StateFlag::REMAPPED) flag = StateFlag::TRIAL;

  // check yield
  double Y = flow_stress(hardening, rate_dep, props, temp, ep, epdot);
  tensor_type S0 = Omega_h::deviator(Te);
  double norm_S0 = Omega_h::norm(S0);
  double f = norm_S0 / sq2 - Y / sq3;

  if (f <= tol1) {
    // Elastic loading
    T = 1. * Te;
    if (flag != StateFlag::REMAPPED) flag = StateFlag::ELASTIC;
  }

  else {

    int conv = 0;
    if (flag != StateFlag::REMAPPED) flag = StateFlag::PLASTIC;
    tensor_type N = S0 / norm_S0;  // Flow direction

    for (int iter=0; iter<100; iter++) {

      // Compute the yield stress
      Y = flow_stress(hardening, rate_dep, props, temp, ep, epdot);

      // Compute g
      double g = norm_S0 - sq23 * Y - twomu * gamma;

      // Compute derivatives of g
      double dydg = dflow_stress(hardening, rate_dep, props, temp, ep, epdot, dtime);
      double dg = -twothird * dydg - twomu;

      // Update dgamma
      double dgamma = -g / dg;
      gamma += dgamma;

      // Update state
      double dep = std::max(sq23 * gamma, 0.0);
      epdot = dep / dtime;
      ep += dep;

      tensor_type S = S0 - twomu * gamma * N;
      f = Omega_h::norm(S) / sq2 - Y / sq3;
      if (f < tol1) {
        conv = 1;
        break;
      }
      else if (std::abs(dgamma) < tol2) {
        conv = 1;
        break;
      }
      else if (iter > 24 && f <= tol1*1000.) {
        // Weaker convergence
        conv = 2;
        break;
      }
#ifdef HYPER_EP_VERBOSE_DEBUG
      std::cout << "Iteration: " << iter + 1 << "\n"
                << "\tROOTJ20: " << Omega_h::norm(S0) << "\n"
                << "\tROOTJ2: " << Omega_h::norm(S) << "\n"
                << "\tep: " << ep << "\n"
                << "\tepdot: " << epdot << "\n"
                << "\tgamma: " << gamma << "\n"
                << "\tg: " << g << "\n"
                << "\tdg: " << dg << "\n\n\n";
#endif
    }

    // Update the stress tensor
    T = Te - twomu * gamma * N;

    if (!conv) {
      return ErrorCode::RADIAL_RETURN_FAILURE;
    }
    else if (conv == 2) {
      // print warning about weaker convergence
    }
  }

  if (flag != StateFlag::ELASTIC) {

    // determine elastic deformation
    double jac = Omega_h::determinant(F);
    auto const Bbe = find_bbe(T, mu);
    tensor_type Be = Bbe * std::pow(jac, 2./3.);
    tensor_type Ve = Omega_h::sqrt_spd(Be);
    Fp = Omega_h::invert(Ve) * F;

    if (flag == StateFlag::REMAPPED) {
      // Correct pressure term
      double p = Omega_h::trace(T);
      const double D1 = 6. * (1. - 2. * Nu) / E;
      p = 2. * jac / D1 * (jac - 1.) - p / 3.;
      for(int i = 0; i < 3; ++i) T(i,i) = p;
    }
  }

  return ErrorCode::SUCCESS;
}

OMEGA_H_INLINE
ErrorCode
linearelasticstress(const std::vector<double>& props,
                    const tensor_type& Fe, const double&,
                    tensor_type& T)
{
  double K = props[0] / (3. * (1. - 2. * props[1]));
  double G = props[0] / 2.0 / (1. + props[1]);
  auto I = identity_matrix<3,3>();
  auto grad_u = Fe - I;
  auto strain = (1.0 / 2.0) * (grad_u + transpose(grad_u));
  auto isotropic_strain = (trace(strain) / 3.) * I;
  auto deviatoric_strain = strain - isotropic_strain;
  T = (3. * K) * isotropic_strain + (2.0 * G) * deviatoric_strain;
  return ErrorCode::SUCCESS;
}

/*
 * Update the stress using Neo-Hookean hyperelasticity
 *
 */
OMEGA_H_INLINE
ErrorCode
hyperstress(const std::vector<double>& props,
            const tensor_type& Fe, const double& jac,
            tensor_type& T)
{

  double E = props[0];
  double Nu = props[1];

  // Jacobian and distortion tensor
  double scale = std::pow(jac, -1./3.);
  tensor_type Fb = scale * Fe;

  // Elastic moduli
  double C10 = E / (4. * (1. + Nu));
  double D1 = 6. * (1. - 2. * Nu) / E;
  double EG = 2. * C10 / jac;

  // Deviatoric left Cauchy-Green deformation tensor
  tensor_type Bb = Fb * Omega_h::transpose(Fb);

  // Deviatoric Cauchy stress
  double TRBb = Omega_h::trace(Bb) / 3.;
  for (int i=0; i<3; ++i) Bb(i,i) -= TRBb;
  T = EG * Bb;

  // Pressure response
  double PR{2. / D1 * (jac - 1.)};
  for (int i=0; i<3; ++i) T(i,i) += PR;

  return ErrorCode::SUCCESS;
}

OMEGA_H_INLINE
ErrorCode
eval(const Elastic& elastic,
     const Hardening& hardening,
     const RateDependence& rate_dep,
     const std::vector<double>& props,
     const double& rho, const tensor_type& F,
     const double& dtime, const double& temp,
     tensor_type& T, double& wave_speed,
     tensor_type& Fp, double& ep, double& epdot)
{
  const double jac = Omega_h::determinant(F);

  {
    // wave speed
    double K = props[0] / 3.0 / (1. - 2. * props[1]);
    double G = props[0] / 2.0 / (1. + props[1]);
    auto plane_wave_modulus = K + (4.0 / 3.0) * G;
    wave_speed = std::sqrt(plane_wave_modulus / rho);
  }

  // Determine the stress predictor.
  tensor_type Te;
  tensor_type Fe = F * Omega_h::invert(Fp);

  ErrorCode err_c = ErrorCode::NOT_SET;

  if (elastic == Elastic::LINEAR_ELASTIC) {
    err_c = linearelasticstress(props, Fe, jac, Te);
  }
  else if (elastic == Elastic::NEO_HOOKEAN) {
    err_c = hyperstress(props, Fe, jac, Te);
  }

  if (err_c != ErrorCode::SUCCESS) {
    return err_c;
  }

  // check yield
  auto flag = StateFlag::TRIAL;
  err_c = radial_return(hardening, rate_dep, props, Te, F, temp,
      dtime, T, Fp, ep, epdot, flag);
  if (err_c != ErrorCode::SUCCESS) {
    return err_c;
  }

  return ErrorCode::SUCCESS;
}

} // HyperEpDetails

template <class Elem>
ModelBase* hyper_ep_factory(Simulation& sim, std::string const& name, Teuchos::ParameterList& pl);

#define LGR_EXPL_INST(Elem) \
extern template ModelBase* hyper_ep_factory<Elem>(Simulation&, std::string const&, Teuchos::ParameterList&);
LGR_EXPL_INST_ELEMS
#undef LGR_EXPL_INST

}

#endif // LGR_HYPER_EP_HPP
