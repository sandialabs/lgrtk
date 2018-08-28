#ifndef LGR_MATERIAL_MODELS_INLINE_HPP
#define LGR_MATERIAL_MODELS_INLINE_HPP

#include "MaterialModels.hpp"
#include "FieldsEnum.hpp"
#include "LGRLambda.hpp"
#include "FieldDB.hpp"
#include <Omega_h_eigen.hpp>

namespace lgr {

template <int SpatialDim>
KOKKOS_INLINE_FUNCTION
int MaterialModelBase<SpatialDim>::getUserID() const { return user_id_; }

template <int SpatialDim>
KOKKOS_INLINE_FUNCTION MaterialModelBase<SpatialDim>::~MaterialModelBase() {}

template <int SpatialDim>
KOKKOS_INLINE_FUNCTION
Omega_h::Matrix<SpatialDim, SpatialDim>
    MaterialModelBase<SpatialDim>::matrixExp_(const Omega_h::Matrix<SpatialDim, SpatialDim> &A) const {
  return matrixFunction_(
      A, KOKKOS_LAMBDA(Scalar x)->Scalar { return ::exp(x); });
}

template <int SpatialDim>
KOKKOS_INLINE_FUNCTION
Omega_h::Matrix<SpatialDim, SpatialDim>
MaterialModelBase<SpatialDim>::matrixLog_(const Omega_h::Matrix<SpatialDim, SpatialDim> &A) const {
  return matrixFunction_(
      A, KOKKOS_LAMBDA(Scalar x)->Scalar { return ::log(x); });
}

template <
    MaterialModelType materialType,
    int               SpatialDim>
KOKKOS_INLINE_FUNCTION
void MaterialModel<materialType, SpatialDim>::waveModuli(int ielem, int state, Scalar moduli[2]) const {
  std::ostringstream message;
  message << "MaterialModel: ";
  message << "material type: " << materialInputSelector(materialType) << ": ";
  message << "invalid material. \n";
  throw std::runtime_error(message.str());
}

template <
    MaterialModelType materialType,
    int               SpatialDim>
KOKKOS_INLINE_FUNCTION
void MaterialModel<materialType, SpatialDim>::hyperelasticCauchyStress(int ielem, int state, double time, double dt) const {
  std::ostringstream message;
  message << "MaterialModel: ";
  message << "material type: " << materialInputSelector(materialType) << ": ";
  message << "invalid material. \n";
  throw std::runtime_error(message.str());
}


template <class Derived, int SpatialDim>
void CRTP_MaterialModelBase<Derived, SpatialDim>::updateElements(const Fields &, int state, double time, double dt) {
  const Derived derivedThis = *(static_cast<Derived *>(this));
  typename Fields::index_array_type &elementID =
      this->userMaterialElementIDs_;
  const typename Fields::state_array_type &planeWaveModulus =
      PlaneWaveModulus<Fields>();
  const typename Fields::state_array_type &bulkModulus =
      BulkModulus<Fields>();
  const typename Fields::array_type &userMatID = UserMatID<Fields>();
  const int                          user_mat_id = this->getUserID();

  /*std::function<void(int)>*/ auto updateEL =
      LAMBDA_EXPRESSION(int linearIndex) {
    //update material model (using pre-computed and stored deformation gradient F, density, energy...)
    const int ielem = elementID(linearIndex);
    derivedThis.hyperelasticCauchyStress(ielem, state, time, dt);
    /*
    moduli[0] = bulk modulus
    moduli[1] = plane wave modulus
    */
    Scalar moduli[2];
    derivedThis.waveModuli(ielem, state, moduli);
    bulkModulus(ielem, state) = moduli[0];
    planeWaveModulus(ielem, state) = moduli[1];
    userMatID(ielem) = user_mat_id;
  };  //end lambda updateEL

  Kokkos::parallel_for(elementID.extent(0), updateEL);
}

//curiously recurring template pattern
template <class Derived, int SpatialDim>
void CRTP_MaterialModelBase<Derived, SpatialDim>::initializeElements(const Fields &) {
  const Derived derivedThis = *(static_cast<Derived *>(this));
  typename Fields::index_array_type &elementID =
      this->userMaterialElementIDs_;
  const typename Fields::state_array_type &planeWaveModulus =
      PlaneWaveModulus<Fields>();
  const typename Fields::state_array_type &bulkModulus =
      BulkModulus<Fields>();
  const typename Fields::array_type &userMatID = UserMatID<Fields>();
  const int                          user_mat_id = this->getUserID();

  /*std::function<void(int)>*/ auto initEL =
      LAMBDA_EXPRESSION(int linearIndex) {
    //compute, AND STORE for later use, the cauchy stress
    const int ielem = elementID(linearIndex);
    double time = 0.0;
    double dt = 0.0;
    derivedThis.hyperelasticCauchyStress(ielem, 0, time, dt);
    derivedThis.hyperelasticCauchyStress(ielem, 1, time, dt);
    /*
    moduli[0] = bulk modulus
    moduli[1] = plane wave modulus
    */
    Scalar moduli[2];
    derivedThis.waveModuli(ielem, 0, moduli);
    bulkModulus(ielem, 0) = bulkModulus(ielem, 1) = moduli[0];
    planeWaveModulus(ielem, 0) = planeWaveModulus(ielem, 1) = moduli[1];
    userMatID(ielem) = user_mat_id;
  };  //end lambda initEL

  Kokkos::parallel_for(elementID.extent(0), initEL);
}

template <int SpatialDim>
template <typename OP>
KOKKOS_INLINE_FUNCTION Omega_h::Matrix<SpatialDim, SpatialDim> MaterialModelBase<SpatialDim>::matrixFunction_(
    const Omega_h::Matrix<SpatialDim, SpatialDim> &A, OP op) const {
  auto decomp = decompose_eigen_jacobi(A, 1e-14);
  for (int i = 0; i < SpatialDim; ++i) decomp.l[i] = op(decomp.l[i]);
  return Omega_h::compose_ortho(decomp.q, decomp.l);
}

}

#endif
