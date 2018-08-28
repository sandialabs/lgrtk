#include "MaterialModels_inline.hpp"
#include "TensorOperations_inline.hpp"

//
// This implementation of Mie Gruneisen is based on the hugoniot
// relations. In the code, you'll see references to 'ph' and 'eh'
// which are the  pressure and energy on the hugoniot, respectively.
//
// The value of 'ph' the hugoniot pressure is found by assuming a
// linear us-up relation:
//
//                     u_s = c_0 + s * u_p
//
// Then, by using the hugoniot equations for the conseration of mass
// and momentum we get to:
//
//                             C_0^2
//   p_h = rho_0 * ------------------------------ * (1 - rho_0 / rho)
//                  (1 - s * (1 - rho_0 / rho))^2
//
// This assumes that the reference pressure is zero and the reference
// density is rho_0.
//
//
//
//
//            HUGONIOT CONSERVATION LAWS
//            --------------------------
//
// Conservation of mass:
// (1)  rho1 * us = rho2 * (us - u2)
// (2)         u2 = us * (1- rho1 / rho2)
//
// Conservation of momentum:
// (3)  p2 - p1 = rho1 * us * u2
//
// Conservation of energy:
// (4)  e2 - e1 = (p2 + p1) * (1 / rho1 - 1 / rho2) / 2
//
// Merging conservation of mass and momentum:
// (5)  p2 - p1 = rho1 * us^2 * (1 - rho1 / rho2)
//
//                    ASSUMPTIONS
//                    -----------
//
// Linear us-up relation - simplified using Equation (2):
// (6)  us = c0 + s * up = c0 + s * u2
// (7)  us = c0 / (1 - s * (1 - rho1 / rho2))
//
// The following substitutions are made to get to the final form:
//   * p2 = ph
//   * p1 = 0
//   * e2 = eh
//   * e1 = 0
//   * rho2 = rho
//   * rho1 = rho0
//
//                   FINAL FORMS
//                   -----------
//
// Reference Hugoniot Pressure - Equation (3) plus Equations (7) and (2)
// (8)  ph = rho0 * us^2 * (1 - rho0 / rho)
// (9)  ph = rho0 * c0^2 / (1 - s * (1 - rho0 / rho))^2 * (1 - rho0 / rho)
//
// Reference Hugoniot Energy - Equation (4)
// (10)  eh = ph * (1 / rho0 - 1 / rho ) / 2
//
// Wrapping it all together, we integrate the Gruneisen model to get and
// explicitly note the dependence of 'ph' and 'eh' on 'rho':
// (11)  p - ph(rho) = Gamma * rho * (e - eh(rho))
// (12)  p = ph(rho) + Gamma * rho * (e - eh(rho))
//

namespace lgr {

//partial template specialization: material model 2: mie gruneisen
template <int SpatialDim>
MaterialModel<MaterialModelType::MIE_GRUNEISEN,SpatialDim>::
MaterialModel(
    int                           arg_user_id,
    Fields &,
    const Teuchos::ParameterList &matData)
    : mass_density(MassDensity<Fields>())
    , internalEnergy(InternalEnergyPerUnitMass<Fields>())
    , stress(Stress<Fields>()) {
  MaterialModelBase<SpatialDim>::setUserID_(
      arg_user_id);
  this->rho0_ = matData.get<double>("rho0");
  this->Gamma0_ = matData.get<double>("Gamma0");
  this->cs_ = matData.get<double>("cs");
  this->s1_ = matData.get<double>("s1");
}

/*
  moduli[0] = bulk modulus
  moduli[1] = plane wave modulus
*/

template <int SpatialDim>
KOKKOS_INLINE_FUNCTION
Scalar
MaterialModel<MaterialModelType::MIE_GRUNEISEN,SpatialDim>::
pressure_(Scalar rho, Scalar e) const {
  const Scalar mu = 1.0 - (rho0_ / rho);

  // For expansion (mu <= 0):
  //  * limit 'us' to not drop below 'cs'

  const Scalar us = cs_ / (1.0 - s1_ * Omega_h::max2(0.0, mu));
  const Scalar ph = rho0_ * us * us * mu;
  const Scalar eh = 0.5 * ph * mu / rho0_;
  const Scalar p = ph + Gamma0_ * rho0_ * (e - eh);
  return p;
}

template <int SpatialDim>
KOKKOS_INLINE_FUNCTION
void
MaterialModel<MaterialModelType::MIE_GRUNEISEN,SpatialDim>::
waveModuli(int ielem, int state, Scalar moduli[2]) const {
  const Scalar rho = mass_density(ielem, state);
  const Scalar e = internalEnergy(ielem, state);
  const Scalar p = this->pressure_(rho, e);

  const Scalar mu = 1.0 - (rho0_ / rho);
  const Scalar dmu =
      rho0_ / rho / rho; /* = \frac{\partial \mu}{\partial \rho} */
  const bool compression = (mu > 0.);

  // For expansion (mu <= 0):
  //  * limit 'us' to not drop below 'cs'
  //  * set 'dus' to '0.0'

  // derivative of pressure with respect to density
  const Scalar us = cs_ / (1.0 - s1_ * Omega_h::max2(0.0, mu));
  const Scalar dus = compression ? (us * s1_ / (1.0 - s1_ * mu)) * dmu : 0.0;
  const Scalar ph = rho0_ * us * us * mu;
  const Scalar dph = 2.0 * rho0_ * us * dus * mu + rho0_ * us * us * dmu;
  const Scalar deh = 0.5 * (mu * dph + ph * dmu) / rho0_;
  const Scalar dpdrho = dph - Gamma0_ * rho0_ * deh;

  //derivative of pressure with repect to energy
  const Scalar dpde = Gamma0_ * rho0_;

  //bulk modulus
  const Scalar K = rho * dpdrho + (p / rho) * dpde;
  moduli[0] = K;
  moduli[1] = K;
}

template <int SpatialDim>
KOKKOS_INLINE_FUNCTION
void
MaterialModel<MaterialModelType::MIE_GRUNEISEN,SpatialDim>::
hyperelasticCauchyStress(int ielem, int state, double /*time*/, double /*dt*/) const {

  const Scalar rho = mass_density(ielem, state);
  const Scalar e = internalEnergy(ielem, state);
  const Scalar p = this->pressure_(rho, e);

  tensorOps::initDiagonal(ielem, state, -p, stress);
}

template struct MaterialModel<MaterialModelType::MIE_GRUNEISEN, 3>;
template struct CRTP_MaterialModelBase<MaterialModel<MaterialModelType::MIE_GRUNEISEN, 3>, 3>;
template struct MaterialModel<MaterialModelType::MIE_GRUNEISEN, 2>;
template struct CRTP_MaterialModelBase<MaterialModel<MaterialModelType::MIE_GRUNEISEN, 2>, 2>;

}
