#include "MaterialModels_inline.hpp"
#include "IdealGasFunctions.hpp"

namespace lgr {

//partial template specialization: material model 1: ideal gas
template <int SpatialDim>
MaterialModel<MaterialModelType::IDEAL_GAS,SpatialDim>::
MaterialModel(
    int                           arg_user_id,
    Fields &,
    const Teuchos::ParameterList &matData)
    : mass_density(MassDensity<Fields>())
    , internalEnergy(InternalEnergyPerUnitMass<Fields>())
    , stress(Stress<Fields>()) {
  MaterialModelBase<SpatialDim>::setUserID_(
      arg_user_id);
  this->gamma_ = matData.get<double>("gamma");
}

/*
  moduli[0] = bulk modulus
  moduli[1] = plane wave modulus
*/
template <int SpatialDim>
KOKKOS_INLINE_FUNCTION
void
MaterialModel<MaterialModelType::IDEAL_GAS,SpatialDim>::
waveModuli(int ielem, int state, Scalar moduli[2]) const {
  const Scalar K = IdealGasFunctions::waveModuli(internalEnergy(ielem, state),
                                                 mass_density(ielem, state),
                                                 gamma_);
  moduli[0] = K;
  moduli[1] = K;
}

template <int SpatialDim>
KOKKOS_INLINE_FUNCTION
void
MaterialModel<MaterialModelType::IDEAL_GAS,SpatialDim>::
hyperelasticCauchyStress(int ielem, int state, double /*time*/, double /*dt*/) const {

  static_assert(SpatialDim == 3 || SpatialDim == 2, "template parameter SpatialDim must be 2 or 3");

  const Scalar p = IdealGasFunctions::hyperelasticCauchyStress(internalEnergy(ielem, state),
                                                               mass_density(ielem, state),
                                                               gamma_);

  const int K_S_XX = FieldsEnum<SpatialDim>::K_S_XX;
  const int K_S_YY = FieldsEnum<SpatialDim>::K_S_YY;
  const int K_S_XY = FieldsEnum<SpatialDim>::K_S_XY;
  stress(ielem, K_S_XX, state) = -p;
  stress(ielem, K_S_YY, state) = -p;
  stress(ielem, K_S_XY, state) = 0.0;

  if(SpatialDim == 3) {
      const int K_S_ZZ = FieldsEnum<3>::K_S_ZZ;
      const int K_S_YZ = FieldsEnum<3>::K_S_YZ;
      const int K_S_ZX = FieldsEnum<3>::K_S_ZX;

      stress(ielem, K_S_ZZ, state) = -p;
      stress(ielem, K_S_YZ, state) = 0.0;
      stress(ielem, K_S_ZX, state) = 0.0;
  }
}
#define LGR_EXPL_INST_DECL(SpatialDim) \
template struct MaterialModel<MaterialModelType::IDEAL_GAS, SpatialDim>; \
template struct CRTP_MaterialModelBase<MaterialModel<MaterialModelType::IDEAL_GAS, SpatialDim>, SpatialDim>;

LGR_EXPL_INST_DECL(3)
LGR_EXPL_INST_DECL(2)
#undef LGR_EXPL_INST_DECL


}
