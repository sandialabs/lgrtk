#ifndef LGR_HYPER_EP_COMMON_HPP
#define LGR_HYPER_EP_COMMON_HPP

#include <hpc_symmetric3x3.hpp>

namespace lgr {
namespace hyper_ep {

enum class ErrorCode {
  NOT_SET,
  SUCCESS,
  LINEAR_ELASTIC_FAILURE,
  HYPERELASTIC_FAILURE,
  RADIAL_RETURN_FAILURE,
  ELASTIC_DEFORMATION_UPDATE_FAILURE,
  MODEL_EVAL_FAILURE
};

enum class Elastic { LINEAR_ELASTIC, NEO_HOOKEAN };

enum class Hardening {
  NONE,
  LINEAR_ISOTROPIC,
  POWER_LAW,
  ZERILLI_ARMSTRONG,
  JOHNSON_COOK
};

enum class RateDependence { NONE, ZERILLI_ARMSTRONG, JOHNSON_COOK };

enum class Damage { NONE, JOHNSON_COOK };

enum class StateFlag { NONE, TRIAL, ELASTIC, PLASTIC, REMAPPED };

struct Properties {
  // Elasticity
  Elastic elastic;
  double E;
  double nu;

  // Plasticity
  Hardening hardening;
  RateDependence rate_dep;
  double A;
  double B;  // Hardening modulus
  double n;  // exponent in hardening
  double C1;
  double C2;
  double C3;
  double C4;
  double eps_dot0;

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
  double D0;
  double DC;
  double eps_f_min;

  Properties()
      : elastic(Elastic::LINEAR_ELASTIC),
        hardening(Hardening::NONE),
        rate_dep(RateDependence::NONE),
        damage(Damage::NONE),
        allow_no_tension(true),
        allow_no_shear(false),
        set_stress_to_zero(false) {}
};

inline char const* get_error_code_string(ErrorCode code) {
  switch (code) {
    case ErrorCode::NOT_SET:
      return "NOT SET";
    case ErrorCode::SUCCESS:
      return "SUCCESS";
    case ErrorCode::LINEAR_ELASTIC_FAILURE:
      return "LINEAR ELASTIC FAILURE";
    case ErrorCode::HYPERELASTIC_FAILURE:
      return "HYPERELASTIC FAILURE";
    case ErrorCode::RADIAL_RETURN_FAILURE:
      return "RADIAL RETURN FAILURE";
    case ErrorCode::ELASTIC_DEFORMATION_UPDATE_FAILURE:
      return "ELASTIC DEFORMATION UPDATE FAILURE";
    case ErrorCode::MODEL_EVAL_FAILURE:
      return "MODEL EVAL FAILURE";
  }
  return "UNKNOWN";
}

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
HPC_NOINLINE inline
hpc::symmetric_deformation<double>
find_bbe(hpc::symmetric_stress<double> const tau, double const mu)
{
  constexpr int maxit = 25;
  constexpr double tol = 1e-12;
  auto const txx = tau(0, 0);
  auto const tyy = tau(1, 1);
  auto const tzz = tau(2, 2);
  auto const txy = .5 * (tau(0, 1) + tau(1, 0));
  auto const txz = .5 * (tau(0, 2) + tau(2, 0));
  auto const tyz = .5 * (tau(1, 2) + tau(2, 1));
  auto Be = deviatoric_part(tau) / mu;
  double bzz_old = 1;
  double bzz_new = 1;
  for (int i = 0; i < maxit; i++) {
    // computes det(BBe), where BBe is the iscohoric deformation
    auto const fun_val =
        (bzz_old * mu *
                (-txy * txy +
                    (bzz_old * mu + txx - tzz) * (bzz_old * mu + tyy - tzz)) +
            2 * txy * txz * tyz + txz * txz * (-bzz_old * mu - tyy + tzz) +
            tyz * tyz * (-bzz_old * mu - txx + tzz)) /
        (mu * mu * mu);
    // computes d(det(BBe) - 1)/d(be_zz), where BBe is the iscohoric deformation
    auto const dfun_val =
        (bzz_old * mu * (2.0 * bzz_old * mu + txx + tyy - 2.0 * tzz) -
            txy * txy - txz * txz - tyz * tyz +
            (bzz_old * mu + txx - tzz) * (bzz_old * mu + tyy - tzz)) /
        (mu * mu);
    bzz_new = bzz_old - (fun_val - 1.0) / dfun_val;
    Be(0, 0) = (1.0 / mu) * (mu * bzz_new + txx - tzz);
    Be(1, 1) = (1.0 / mu) * (mu * bzz_new + tyy - tzz);
    Be(2, 2) = bzz_new;
    if ((bzz_new - bzz_old) * (bzz_new - bzz_old) < tol) {
      return Be;
    }
    bzz_old = bzz_new;
  }
  assert(false);
  hpc::symmetric_deformation<double>();
}


}  // namespace hyper_ep
}  // namespace lgr

#endif  // LGR_HYPER_EP_COMMON_HPP
