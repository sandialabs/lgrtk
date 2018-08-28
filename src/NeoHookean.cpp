#include "MaterialModels_inline.hpp"

#include "FieldDB.hpp"
#include "TensorOperations_inline.hpp"

namespace lgr {

// Neo-Hookean model
/*
  equations (2.37), (2.39), (2.40) of
  @article{simo1992associative,
  title={Associative coupled thermoplasticity at finite strains: formulation, numerical analysis and implementation},
  author={Simo, JC and Miehe, Ch},
  journal={Computer Methods in Applied Mechanics and Engineering},
  volume={98},
  number={1},
  pages={41--104},
  year={1992},
  publisher={Elsevier}
  }
  except that
  c0 <-- rho0*c0
*/

template <int SpatialDim>
MaterialModel<MaterialModelType::NEO_HOOKEAN,SpatialDim>::
MaterialModel(
    int                           arg_user_id,
    Fields &,
    const Teuchos::ParameterList &matData)
    : F(DeformationGradient<Fields>())
    , mass_density(MassDensity<Fields>())
    , internalEnergy(InternalEnergyPerUnitMass<Fields>())
    , stress(Stress<Fields>())
    , userMatID(UserMatID<Fields>())
    , c0(1.0)
    , beta(0.0)
    , theta0(298.0)
    , p0(0.0) {
  MaterialModelBase<SpatialDim>::setUserID_(
      arg_user_id);
  const Scalar E = matData.get<double>("Youngs Modulus");
  const Scalar nu = matData.get<double>("Poissons Ratio");
  this->shear_modulus = E / 2.0 / (1. + nu);
  this->bulk_modulus = E / (3. * (1. - 2. * nu));

  const double *pc0 = matData.getPtr<double>("specific heat per unit mass");
  if (0 != pc0) c0 = (*pc0);

  const double *pbeta =
      matData.getPtr<double>("coefficient of thermal expansion");
  if (0 != pbeta) beta = (*pbeta);

  const double *ptheta0 = matData.getPtr<double>("reference temperature");
  if (0 != ptheta0) theta0 = (*ptheta0);

  const double *pp0 = matData.getPtr<double>("undeformed pressure");
  if (0 != pp0) p0 = (*pp0);
}

template <int SpatialDim>
KOKKOS_INLINE_FUNCTION
Scalar
MaterialModel<MaterialModelType::NEO_HOOKEAN,SpatialDim>::
U(Scalar J) const {
  return bulk_modulus * (0.5 * (J * J - 1.0) - log(J));
}

template <int SpatialDim>
KOKKOS_INLINE_FUNCTION
Scalar
MaterialModel<MaterialModelType::NEO_HOOKEAN,SpatialDim>::
Uprime(Scalar J) const { return bulk_modulus * (J - (1.0 / J)); }

template <int SpatialDim>
KOKKOS_INLINE_FUNCTION
Scalar
MaterialModel<MaterialModelType::NEO_HOOKEAN,SpatialDim>::
Uprime2(Scalar J) const { return bulk_modulus * (1 + (1.0 / J / J)); }

template <int SpatialDim>
KOKKOS_INLINE_FUNCTION
Scalar
MaterialModel<MaterialModelType::NEO_HOOKEAN,SpatialDim>::
Uprime3(Scalar J) const { return bulk_modulus * (-2.0 / J / J / J); }

template <int SpatialDim>
KOKKOS_INLINE_FUNCTION
Scalar
MaterialModel<MaterialModelType::NEO_HOOKEAN,SpatialDim>::
M(Scalar J, Scalar theta) const {
  return -3.0 * beta * (theta - theta0) * Uprime(J);
}

template <int SpatialDim>
KOKKOS_INLINE_FUNCTION
Scalar
MaterialModel<MaterialModelType::NEO_HOOKEAN,SpatialDim>::
DM_DTheta(Scalar J, Scalar) const {
  return -3.0 * beta * Uprime(J);
}

template <int SpatialDim>
KOKKOS_INLINE_FUNCTION
Scalar
MaterialModel<MaterialModelType::NEO_HOOKEAN,SpatialDim>::
D2M_DTheta2(Scalar, Scalar) const { return 0.0; }

template <int SpatialDim>
KOKKOS_INLINE_FUNCTION
Scalar
MaterialModel<MaterialModelType::NEO_HOOKEAN,SpatialDim>::
DM_DJ(Scalar J, Scalar theta) const {
  return -3.0 * beta * (theta - theta0) * Uprime2(J);
}

template <int SpatialDim>
KOKKOS_INLINE_FUNCTION
Scalar
MaterialModel<MaterialModelType::NEO_HOOKEAN,SpatialDim>::
D2M_DJ_DTheta(Scalar J, Scalar) const {
  return -3.0 * beta * Uprime2(J);
}

template <int SpatialDim>
KOKKOS_INLINE_FUNCTION
Scalar
MaterialModel<MaterialModelType::NEO_HOOKEAN,SpatialDim>::
D2M_DJ2(Scalar J, Scalar theta) const {
  return -3.0 * beta * (theta - theta0) * Uprime3(J);
}

template <int SpatialDim>
KOKKOS_INLINE_FUNCTION
Scalar
MaterialModel<MaterialModelType::NEO_HOOKEAN,SpatialDim>::
T(Scalar rho0, Scalar theta) const {
  return rho0 * c0 * ((theta - theta0) - theta * log(theta / theta0));
}

template <int SpatialDim>
KOKKOS_INLINE_FUNCTION
Scalar
MaterialModel<MaterialModelType::NEO_HOOKEAN,SpatialDim>::
Tprime(Scalar rho0, Scalar theta) const {
  return -rho0 * c0 * log(theta / theta0);
}

template <int SpatialDim>
KOKKOS_INLINE_FUNCTION
Scalar
MaterialModel<MaterialModelType::NEO_HOOKEAN,SpatialDim>::
Tprime2(Scalar rho0, Scalar theta) const { return -rho0 * c0 / theta; }

template <int SpatialDim>
KOKKOS_INLINE_FUNCTION
Scalar
MaterialModel<MaterialModelType::NEO_HOOKEAN,SpatialDim>::
W(const Omega_h::Matrix<SpatialDim, SpatialDim> &elem_F, Scalar J) const {
  Scalar e = inner_product(elem_F, elem_F);
  const Scalar J3 = cbrt(J);
  const Scalar J23 = J3 * J3;
  const Scalar Jminus23 = 1.0 / J23;
  e *= Jminus23;
  e -= 3.0;
  const Scalar muOver2 = 0.5 * (this->shear_modulus);
  e *= muOver2;
  return e;
}

template <int SpatialDim>
KOKKOS_INLINE_FUNCTION
Scalar
MaterialModel<MaterialModelType::NEO_HOOKEAN,SpatialDim>::
eta(Scalar J, Scalar rho0, Scalar theta) const {
  return -(DM_DTheta(J, theta) + Tprime(rho0, theta));
}

template <int SpatialDim>
KOKKOS_INLINE_FUNCTION
Scalar
MaterialModel<MaterialModelType::NEO_HOOKEAN,SpatialDim>::
Deta_DTheta(Scalar J, Scalar rho0, Scalar theta) const {
  return -(D2M_DTheta2(J, theta) + Tprime2(rho0, theta));
}

template <int SpatialDim>
KOKKOS_INLINE_FUNCTION
Scalar
MaterialModel<MaterialModelType::NEO_HOOKEAN,SpatialDim>::
temperature(
    const Omega_h::Matrix<SpatialDim, SpatialDim> &elem_F, Scalar J, Scalar rho, Scalar e) const {
  const Scalar rho0 = rho * J;
  const Scalar DEVE = W(elem_F, J);
  const Scalar VOLE = U(J);
  const Scalar E = rho0 * e;
  /*
    for this specific material model, the equation for temperature theta
    can be algebraically reduced to a linear equation. (all the log() terms
    cancel.)  in fact, using mathematica (or a lot of algebra on paper), one can
    derive:

    rho0*e = U + W + rho0*c0*theta + theta0*( 3*(J*J-1)*beta*bulk_modulus/J - rho0*c0 )

    however, in solving this for temperature theta, theta can become negative.

    for more general material models, the equivalent equation for temperature is not linear.
    in that case, a newton iteration is probably necessary.

    it is not possible to take the log of a negative number; this can mess up the second
    iteration of newton loop.  for now, do only one newton iteration to avoid that problem.
  */
  Scalar theta = theta0;
  for (int iter = 0; iter < 1; ++iter) {
    const Scalar entropy = eta(J, rho0, theta);
    const Scalar R =
        DEVE + VOLE + M(J, theta) + T(rho0, theta) + theta * entropy - E;
    const Scalar Rprime = DM_DTheta(J, theta) + Tprime(rho0, theta) +
                          entropy + theta * Deta_DTheta(J, rho0, theta);
    theta -= (R / Rprime);
  }
  return theta;
}

/*
  moduli[0] = bulk modulus
  moduli[1] = plane wave modulus
*/
template <int SpatialDim>
KOKKOS_INLINE_FUNCTION
void
MaterialModel<MaterialModelType::NEO_HOOKEAN,SpatialDim>::
waveModuli(int ielem, int state, Scalar moduli[2]) const {

  const Omega_h::Matrix<SpatialDim, SpatialDim> elemF =
          tensorOps::fillMatrix<Omega_h::Matrix<SpatialDim, SpatialDim>>(F, ielem);
  const Scalar J = Omega_h::determinant(elemF);

  const Scalar e = internalEnergy(ielem, state);
  const Scalar rho = mass_density(ielem, state);
  const Scalar rho0 = rho * J;
  const Scalar theta = temperature(elemF, J, rho, e);

  Scalar DTheta_DJ = D2M_DJ_DTheta(J, theta);
  DTheta_DJ /= (-D2M_DTheta2(J, theta) - Tprime2(rho0, theta));

  Scalar tangentBulkModulus = Uprime2(J);
  tangentBulkModulus += D2M_DJ2(J, theta);
  tangentBulkModulus += DTheta_DJ * D2M_DJ_DTheta(J, theta);
  tangentBulkModulus *= J;

  moduli[0] = tangentBulkModulus;

  const double planeWaveModulus =
      tangentBulkModulus + (4.0 / 3.0) * shear_modulus;
  moduli[1] = planeWaveModulus;
}

template <int SpatialDim>
KOKKOS_INLINE_FUNCTION
void
MaterialModel<MaterialModelType::NEO_HOOKEAN,SpatialDim>::
hyperelasticCauchyStress(int ielem, int state, double /*time*/, double /*dt*/) const {

  const Omega_h::Matrix<SpatialDim, SpatialDim> elemF =
          tensorOps::fillMatrix<Omega_h::Matrix<SpatialDim, SpatialDim>>(F, ielem);

  const Scalar J = Omega_h::determinant(elemF);

  Scalar pressure = p0;

  //mechanical volumentric term
  pressure -= Uprime(J);

  //thermal expansion term
  const Scalar e = internalEnergy(ielem, state);
  const Scalar rho = mass_density(ielem, state);
  const Scalar theta = temperature(elemF, J, rho, e);
  pressure -= DM_DJ(J, theta);

  //mechanical deviatoric term
  const Scalar J3 = cbrt(J);
  const Scalar J23 = J3 * J3;
  const Scalar Jminus23 = 1.0 / J23;
  Omega_h::Matrix<SpatialDim, SpatialDim> btilde = elemF * Omega_h::transpose(elemF);
  btilde *= Jminus23;
  const Scalar          muOverJ = shear_modulus / J;
  Omega_h::Matrix<SpatialDim, SpatialDim> sigma = Omega_h::deviator(btilde);
  sigma *= muOverJ;

  //assign final pressure term
  for(int d = 0; d < SpatialDim; ++d)
      sigma(d, d) -= pressure;

  static_assert(SpatialDim == 2 || SpatialDim == 3, "SpatialDim must be 2 or 3");

  if(SpatialDim == 2) {
      stress(ielem, FieldsEnum<2>::K_S_XX, state) = sigma(0,0);
      stress(ielem, FieldsEnum<2>::K_S_YY, state) = sigma(1,1);
      stress(ielem, FieldsEnum<2>::K_S_XY, state) = 0.5*(sigma(0,1) + sigma(1,0));
  } else {
      stress(ielem, FieldsEnum<3>::K_S_XX, state) = sigma(0,0);
      stress(ielem, FieldsEnum<3>::K_S_YY, state) = sigma(1,1);
      stress(ielem, FieldsEnum<3>::K_S_ZZ, state) = sigma(2,2);
      stress(ielem, FieldsEnum<3>::K_S_XY, state) = 0.5*(sigma(0,1) + sigma(1,0));
      stress(ielem, FieldsEnum<3>::K_S_XZ, state) = 0.5*(sigma(0,2) + sigma(2,0));
      stress(ielem, FieldsEnum<3>::K_S_YZ, state) = 0.5*(sigma(1,2) + sigma(2,1));
  }

}

template struct MaterialModel<MaterialModelType::NEO_HOOKEAN, 3>;
template struct CRTP_MaterialModelBase<MaterialModel<MaterialModelType::NEO_HOOKEAN, 3>, 3>;
template struct MaterialModel<MaterialModelType::NEO_HOOKEAN, 2>;
template struct CRTP_MaterialModelBase<MaterialModel<MaterialModelType::NEO_HOOKEAN, 2>, 2>;

}
