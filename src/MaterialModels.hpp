/*
//@HEADER
// ************************************************************************
//
//                        lgr v. 1.0
//              Copyright (2014) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact  Glen A. Hansen (gahanse@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#ifndef LGR_MATERIAL_MODELS_HPP
#define LGR_MATERIAL_MODELS_HPP

#include <list>
#include <string>
#include "Fields.hpp"
#include <Omega_h_assoc.hpp>
#include <Omega_h_matrix.hpp>

namespace lgr {

enum class MaterialModelType {
  NEO_HOOKEAN = 0,
  IDEAL_GAS,
  MIE_GRUNEISEN,
  J2_PLASTICITY,
  NUMBER_OF_MATERIAL_MODELS
};

MaterialModelType materialInputSelector(
    const std::string &materialModelString);
std::string materialInputSelector(const MaterialModelType type);

template <int SpatialDim>
class MaterialModelBase {
 public:
  typedef typename lgr::Fields<SpatialDim> Fields;

 private:
  int user_id_;

 protected:
  void setUserID_(int arg_user_id);

  //store list of element numbers associated with this material.
  typename Fields::index_array_type userMaterialElementIDs_;

  template <typename OP>
  KOKKOS_INLINE_FUNCTION Omega_h::Matrix<SpatialDim, SpatialDim> matrixFunction_(
      const Omega_h::Matrix<SpatialDim, SpatialDim> &A, OP op) const;

 public:
  MaterialModelBase();
  MaterialModelBase(int arg_user_id);

  KOKKOS_INLINE_FUNCTION
  int getUserID() const;

  virtual void initializeElements(const Fields &mesh_fields) = 0;
  virtual void updateElements(const Fields &mesh_fields, int state, double time, double dt) = 0;
  KOKKOS_INLINE_FUNCTION virtual ~MaterialModelBase();

  void createElementIDs(
      const Teuchos::Array<std::string> &elementBlocks,
      const Omega_h::MeshDimSets &       elementsets);

  typename Fields::index_array_type &getUserMaterialElementIDs();

  KOKKOS_INLINE_FUNCTION
  Omega_h::Matrix<SpatialDim, SpatialDim>
      matrixExp_(const Omega_h::Matrix<SpatialDim, SpatialDim> &A) const;

  KOKKOS_INLINE_FUNCTION
  Omega_h::Matrix<SpatialDim, SpatialDim>
      matrixLog_(const Omega_h::Matrix<SpatialDim, SpatialDim> &A) const;
};

template <
    MaterialModelType materialType,
    int               SpatialDim>
struct MaterialModel
    : public MaterialModelBase<SpatialDim> {
  typedef lgr::Fields<SpatialDim> Fields;

  /*
    moduli[0] = bulk modulus
    moduli[1] = plane wave modulus
  */
  KOKKOS_INLINE_FUNCTION
  void waveModuli(int ielem, int state, Scalar moduli[2]) const;
  KOKKOS_INLINE_FUNCTION
  void hyperelasticCauchyStress(int ielem, int state, double time, double dt) const;
  void initializeElements(const Fields &mesh_fields) override;
  void updateElements(const Fields &mesh_fields, int state, double time, double dt) override;
};

//curiously recurring template pattern
template <
    class Derived,
    int SpatialDim>
struct CRTP_MaterialModelBase
    : public MaterialModelBase<SpatialDim> {
  typedef lgr::Fields<SpatialDim> Fields;
  typedef typename lgr::MaterialModelBase<SpatialDim>
      MaterialModelBase;

  void initializeElements(const Fields &mesh_fields) override;

  void updateElements(const Fields &mesh_fields, int state, double time, double dt) override;
};

//partial template specialization: material model 0: neo hookean elastic
template <int SpatialDim>
struct MaterialModel<
    MaterialModelType::NEO_HOOKEAN,
    SpatialDim>
    : public CRTP_MaterialModelBase<
          MaterialModel<
              MaterialModelType::NEO_HOOKEAN,
              SpatialDim>,
          SpatialDim> {
  typedef lgr::Fields<SpatialDim> Fields;

  Scalar shear_modulus;
  Scalar bulk_modulus;

  const typename Fields::elem_tensor_type           F;
  const typename Fields::state_array_type           mass_density;
  const typename Fields::state_array_type           internalEnergy;
  const typename Fields::elem_sym_tensor_state_type stress;
  const typename Fields::array_type                 userMatID;

  Scalar c0;      //specific heat per unit mass
  Scalar beta;    //coefficient of thermal expansion
  Scalar theta0;  //reference temperature
  Scalar p0;      //un-deformed pressure

  MaterialModel(
      int                           arg_user_id,
      Fields &                      meshFields,
      const Teuchos::ParameterList &matData);

  KOKKOS_INLINE_FUNCTION
  Scalar U(Scalar J) const;

  KOKKOS_INLINE_FUNCTION
  Scalar Uprime(Scalar J) const;

  KOKKOS_INLINE_FUNCTION
  Scalar Uprime2(Scalar J) const;

  KOKKOS_INLINE_FUNCTION
  Scalar Uprime3(Scalar J) const;

  KOKKOS_INLINE_FUNCTION
  Scalar M(Scalar J, Scalar theta) const;

  KOKKOS_INLINE_FUNCTION
  Scalar DM_DTheta(Scalar J, Scalar theta) const;

  KOKKOS_INLINE_FUNCTION
  Scalar D2M_DTheta2(Scalar J, Scalar theta) const;

  KOKKOS_INLINE_FUNCTION
  Scalar DM_DJ(Scalar J, Scalar theta) const;

  KOKKOS_INLINE_FUNCTION
  Scalar D2M_DJ_DTheta(Scalar J, Scalar theta) const;

  KOKKOS_INLINE_FUNCTION
  Scalar D2M_DJ2(Scalar J, Scalar theta) const;

  KOKKOS_INLINE_FUNCTION
  Scalar T(Scalar rho0, Scalar theta) const;

  KOKKOS_INLINE_FUNCTION
  Scalar Tprime(Scalar rho0, Scalar theta) const;

  KOKKOS_INLINE_FUNCTION
  Scalar Tprime2(Scalar rho0, Scalar theta) const;

  KOKKOS_INLINE_FUNCTION
  Scalar W(const Omega_h::Matrix<SpatialDim, SpatialDim> &F, Scalar J) const;

  KOKKOS_INLINE_FUNCTION
  Scalar eta(Scalar J, Scalar rho0, Scalar theta) const;

  KOKKOS_INLINE_FUNCTION
  Scalar Deta_DTheta(Scalar J, Scalar rho0, Scalar theta) const;

  KOKKOS_INLINE_FUNCTION
  Scalar temperature(
      const Omega_h::Matrix<SpatialDim, SpatialDim> &F, Scalar J, Scalar rho, Scalar e) const;

  /*
    moduli[0] = bulk modulus
    moduli[1] = plane wave modulus
  */
  KOKKOS_INLINE_FUNCTION
  void waveModuli(int ielem, int state, Scalar moduli[2]) const;

  KOKKOS_INLINE_FUNCTION
  void hyperelasticCauchyStress(int ielem, int state, double time, double dt) const;
};

//partial template specialization: material model 1: ideal gas
template <int SpatialDim>
struct MaterialModel<
    MaterialModelType::IDEAL_GAS,
    SpatialDim>
    : public CRTP_MaterialModelBase<
          MaterialModel<
              MaterialModelType::IDEAL_GAS,
              SpatialDim>,
          SpatialDim> {
  typedef lgr::Fields<SpatialDim> Fields;

  Scalar                                            gamma_;
  const typename Fields::state_array_type           mass_density;
  const typename Fields::state_array_type           internalEnergy;
  const typename Fields::elem_sym_tensor_state_type stress;

  MaterialModel(
      int                           arg_user_id,
      Fields &                      meshFields,
      const Teuchos::ParameterList &matData);

  /*
    moduli[0] = bulk modulus
    moduli[1] = plane wave modulus
  */
  KOKKOS_INLINE_FUNCTION
  void waveModuli(int ielem, int state, Scalar moduli[2]) const;

  KOKKOS_INLINE_FUNCTION
  void hyperelasticCauchyStress(int ielem, int state, double time, double dt) const;
};

//partial template specialization: material model 2: mie gruneisen
template <int SpatialDim>
struct MaterialModel<
    MaterialModelType::MIE_GRUNEISEN,
    SpatialDim>
    : public CRTP_MaterialModelBase<
          MaterialModel<
              MaterialModelType::MIE_GRUNEISEN,
              SpatialDim>,
          SpatialDim> {
  typedef lgr::Fields<SpatialDim> Fields;

  Scalar rho0_;
  Scalar Gamma0_;
  Scalar cs_;
  Scalar s1_;

  const typename Fields::state_array_type           mass_density;
  const typename Fields::state_array_type           internalEnergy;
  const typename Fields::elem_sym_tensor_state_type stress;

  MaterialModel(
      int                           arg_user_id,
      Fields &                      meshFields,
      const Teuchos::ParameterList &matData);

  /*
    moduli[0] = bulk modulus
    moduli[1] = plane wave modulus
  */

  KOKKOS_INLINE_FUNCTION
  double pressure_(Scalar rho, Scalar e) const;

  KOKKOS_INLINE_FUNCTION
  void waveModuli(int ielem, int state, Scalar moduli[2]) const;

  KOKKOS_INLINE_FUNCTION
  void hyperelasticCauchyStress(int ielem, int state, double time, double dt) const;
};


//partial template specialization: material model: J2 Plasticity
template <int SpatialDim>
struct MaterialModel<
    MaterialModelType::J2_PLASTICITY,
    SpatialDim>
    : public CRTP_MaterialModelBase<
          MaterialModel<
              MaterialModelType::J2_PLASTICITY,
              SpatialDim>,
          SpatialDim> {
  typedef typename lgr::Fields<SpatialDim> Fields;
  typedef Kokkos::View<Omega_h::Vector< Fields::SymTensorLength> * [Fields::NumStates], ExecSpace> Gp_type;
  typedef Kokkos::View<Scalar * [Fields::NumStates], ExecSpace> xi_type;

  const typename Fields::elem_tensor_type           F;
  const typename Fields::elem_sym_tensor_state_type stress;

  Scalar bulkModulus_;
  Scalar shearModulus_;
  Scalar yieldStress_;
  Scalar hardeningModulus_;
  Scalar p0_;  //undeformed pressure

  Gp_type Gp_;
  xi_type xi_;

  MaterialModel(
      int                           arg_user_id,
      Fields &                      meshFields,
      const Teuchos::ParameterList &matData);

  /*
    moduli[0] = bulk modulus
    moduli[1] = plane wave modulus
  */
  KOKKOS_INLINE_FUNCTION
  void waveModuli(int ielem, int state, Scalar moduli[2]) const;

  KOKKOS_INLINE_FUNCTION
  void hyperelasticCauchyStress(int ielem, int commitState, double time, double dt) const;
};

template <int SpatialDim>
std::shared_ptr<MaterialModelBase<SpatialDim>>
createMaterialModel(
    int                                                 arg_user_material_id,
    const MaterialModelType                             matModelType,
    lgr::Fields<SpatialDim> &mesh_fields,
    const Teuchos::ParameterList &                      matData);

template <int SpatialDim>
void createMaterialModels(
    const Teuchos::ParameterList &materialModelParameterList,
    lgr::Fields<SpatialDim> &mesh_fields,
    const Omega_h::MeshDimSets &                        elementSets,
    std::list<
        std::shared_ptr<MaterialModelBase<SpatialDim>>>
        &theMaterialModels);

#define LGR_EXPL_INST_DECL(SpatialDim) \
extern template class MaterialModelBase<SpatialDim>; \
extern template struct MaterialModel<MaterialModelType::NEO_HOOKEAN, SpatialDim>; \
extern template struct MaterialModel<MaterialModelType::IDEAL_GAS, SpatialDim>; \
extern template struct MaterialModel<MaterialModelType::MIE_GRUNEISEN, SpatialDim>; \
extern template struct MaterialModel<MaterialModelType::J2_PLASTICITY, SpatialDim>; \
extern template struct CRTP_MaterialModelBase<MaterialModel<MaterialModelType::NEO_HOOKEAN, SpatialDim>, SpatialDim>; \
extern template struct CRTP_MaterialModelBase<MaterialModel<MaterialModelType::IDEAL_GAS, SpatialDim>, SpatialDim>; \
extern template struct CRTP_MaterialModelBase<MaterialModel<MaterialModelType::MIE_GRUNEISEN, SpatialDim>, SpatialDim>; \
extern template struct CRTP_MaterialModelBase<MaterialModel<MaterialModelType::J2_PLASTICITY, SpatialDim>, SpatialDim>; \
extern template std::shared_ptr<MaterialModelBase<SpatialDim>> \
createMaterialModel( \
    int                                                 arg_user_material_id, \
    const MaterialModelType                             matModelType, \
    lgr::Fields<SpatialDim> &mesh_fields, \
    const Teuchos::ParameterList &                      matData); \
extern template void createMaterialModels( \
    const Teuchos::ParameterList &materialModelParameterList, \
    lgr::Fields<SpatialDim> &mesh_fields, \
    const Omega_h::MeshDimSets &                        elementSets, \
    std::list< \
        std::shared_ptr<MaterialModelBase<SpatialDim>>> \
        &theMaterialModels);
LGR_EXPL_INST_DECL(3)
LGR_EXPL_INST_DECL(2)
#undef LGR_EXPL_INST_DECL

}  //end namespace lgr

#endif  //MATERIAL_MODELS_HPP
