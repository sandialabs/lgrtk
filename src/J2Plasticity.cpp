#include "MaterialModels_inline.hpp"
#include "TensorOperations_inline.hpp"

namespace lgr {

//partial template specialization: material model: J2 Plasticity
template <int SpatialDim>
MaterialModel<MaterialModelType::J2_PLASTICITY,SpatialDim>::
MaterialModel(
    int                           arg_user_id,
    Fields &                      meshFields,
    const Teuchos::ParameterList &matData)
    : F(DeformationGradient<Fields>())
    , stress(Stress<Fields>())
    , p0_(0.0)
    , Gp_("plastic metric tensor", meshFields.femesh.nelems)
    , xi_("equivalent plastic strain", meshFields.femesh.nelems) {
  FieldDB<Gp_type>::Self()["plastic metric tensor"] = Gp_;
  FieldDB<xi_type>::Self()["equivalent plastic strain"] = xi_;
  MaterialModelBase<SpatialDim>::setUserID_(
      arg_user_id);

  const Scalar E = matData.get<double>("Youngs Modulus");
  const Scalar nu = matData.get<double>("Poissons Ratio");
  this->bulkModulus_ = E / (3. * (1. - 2. * nu));
  this->shearModulus_ = E / 2.0 / (1. + nu);

  this->yieldStress_ = matData.get<double>("Yield Stress");
  this->hardeningModulus_ = matData.get<double>("Hardening Modulus");

  const double *pp0 = matData.getPtr<double>("Undeformed Pressure");
  if (0 != pp0) p0_ = (*pp0);

  const Omega_h::Vector<Fields::SymTensorLength> I(
      Omega_h::symm2vector(Omega_h::identity_matrix<SpatialDim, SpatialDim>()));
  Kokkos::deep_copy(Gp_, I);
}

/*
  moduli[0] = bulk modulus
  moduli[1] = plane wave modulus
*/
template <int SpatialDim>
KOKKOS_INLINE_FUNCTION
void
MaterialModel<MaterialModelType::J2_PLASTICITY,SpatialDim>::
waveModuli(int ielem, int /*state*/, Scalar moduli[2]) const {


  const Omega_h::Matrix<SpatialDim, SpatialDim> elemF =
          tensorOps::fillMatrix<Omega_h::Matrix<SpatialDim, SpatialDim>>(F, ielem);
  const Scalar                J = Omega_h::determinant(elemF);
  const Scalar                K = bulkModulus_ * J * (1. + (1. / J / J));
  moduli[0] = K;
  moduli[1] = K + (4. * shearModulus_ / 3.);
  return;
}

template <int SpatialDim>
KOKKOS_INLINE_FUNCTION
void
MaterialModel<MaterialModelType::J2_PLASTICITY,SpatialDim>::
hyperelasticCauchyStress(int ielem, int commitState, double /*time*/, double /*dt*/) const {
  const int state1 = commitState;
  const int state0 = (commitState + 1) % 2;

  typedef Omega_h::Matrix<SpatialDim, SpatialDim> Tensor;

  constexpr Scalar almostOne(1.0 - DBL_EPSILON);
  constexpr Scalar sq23_0(0.816496580927726);
  constexpr Scalar sq23(
      sq23_0 -
      (sq23_0 * sq23_0 - (2. / 3.)) / (2. * sq23_0));  // std::sqrt(2./3.));

  const Scalar kappa = bulkModulus_;
  const Scalar mu = shearModulus_;
  const Scalar Y = yieldStress_;
  const Scalar H = hardeningModulus_;

  const Tensor Gp_n( Omega_h::vector2symm( Gp_(ielem, state0) ) );
  Tensor       Gp_np1(Gp_n);
  const Scalar xi_n = xi_(ielem, state0);


  const Tensor elemF = tensorOps::fillMatrix<Tensor>(F, ielem);

  //trial elastic state (deviatoric kirchhoff stress)
  const Tensor betr = elemF * Gp_n * Omega_h::transpose(elemF);
  const Tensor eps = 0.5 * (this->matrixLog_(betr));
  Tensor       s = (2. * mu) * Omega_h::deviator(eps);

  //radial return to yield surface
  const Scalar smag = Omega_h::norm(s);
  const Scalar phitr = smag - sq23 * (Y + (H * xi_n));
  if (phitr > 0.0) {
    Scalar dgamma = phitr / ((2. * mu) + (2. * H / 3.));
    dgamma *= almostOne; /*modify to remain slightly outside yield surface*/
    const Tensor n = s / smag;
    s -= ((2. * mu) * dgamma) * n;

    //update isotropic hardening variable
    Scalar &xi_np1 = xi_(ielem, state1);
    xi_np1 = xi_n + sq23 * dgamma;

    //exponential map to get Gp_np1
    const Tensor A = (-2.0 * dgamma) * n;
    const Tensor expA = this->matrixExp_(A);
    const Tensor Finv = Omega_h::invert(elemF);
    Gp_np1 = (Finv * expA * elemF) * Gp_n;
  }

  //update plastic metric tensor
  Gp_(ielem, state1) = Omega_h::symm2vector(Gp_np1);

  //compute pressure
  const Scalar J = Omega_h::determinant(elemF);
  const Scalar p = +p0_ - 0.5 * kappa * (J - 1. / J);

  //final cauchy stress
  Tensor sigma = s;
  sigma /= J;
  for(int i = 0; i < SpatialDim; ++i) {
      sigma(i,i) -= p;
  }
  static_assert(SpatialDim == 2 || SpatialDim == 3, "SpatialDim must be 2 or 3");

  auto stressSubView = Kokkos::subview(stress, ielem, Kokkos::ALL(), state1);
  if(SpatialDim == 2) {
       stressSubView(FieldsEnum<2>::K_S_XX) = sigma(0,0);
       stressSubView(FieldsEnum<2>::K_S_YY) = sigma(1,1);
       stressSubView(FieldsEnum<2>::K_S_XY) = 0.5*(sigma(0,1) + sigma(1,0));
   } else if(SpatialDim == 3){
       stressSubView(FieldsEnum<3>::K_S_XX) = sigma(0,0);
       stressSubView(FieldsEnum<3>::K_S_YY) = sigma(1,1);
       stressSubView(FieldsEnum<3>::K_S_ZZ) = sigma(2,2);
       stressSubView(FieldsEnum<3>::K_S_XY) = 0.5*(sigma(0,1) + sigma(1,0));
       stressSubView(FieldsEnum<3>::K_S_YZ) = 0.5*(sigma(1,2) + sigma(2,1));
       stressSubView(FieldsEnum<3>::K_S_ZX) = 0.5*(sigma(2,0) + sigma(0,2));
   }
}

template struct MaterialModel<MaterialModelType::J2_PLASTICITY, 3>;
template struct CRTP_MaterialModelBase<MaterialModel<MaterialModelType::J2_PLASTICITY, 3>, 3>;
template struct MaterialModel<MaterialModelType::J2_PLASTICITY, 2>;
template struct CRTP_MaterialModelBase<MaterialModel<MaterialModelType::J2_PLASTICITY, 2>, 2>;

}
