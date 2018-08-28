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
// documentation and/or other Conductivitys provided with the distribution.
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

#ifndef LGR_CONDUCTIVITY_MODELS_HPP
#define LGR_CONDUCTIVITY_MODELS_HPP

#include "Fields.hpp"
#include <Omega_h_assoc.hpp>
#include <list>

namespace lgr {

enum class ConductivityModelType {
  CONSTANT = 0,
  NUMBER_OF_CONDUCTIVITY_MODELS
};

ConductivityModelType ConductivityInputSelector(
    const std::string &ConductivityModelString);

std::string ConductivityInputSelector(const ConductivityModelType type);

template <int SpatialDim>
class ConductivityModelBase {
 public:
  typedef typename lgr::Fields<SpatialDim> Fields;

 private:
  int user_id_;

 protected:
  void setUserID_(int arg_user_id) { user_id_ = arg_user_id; }

  //store list of element numbers associated with this Conductivity Model.
  typename Fields::index_array_type userConductivityElementIDs_;

 public:
  ConductivityModelBase();
  ConductivityModelBase(int arg_user_id);

  KOKKOS_INLINE_FUNCTION
  int getUserID() const { return user_id_; }

  virtual void initializeElements(const Fields &mesh_fields) = 0;
  virtual void updateElements(const Fields &mesh_fields, int state) = 0;

  void createElementIDs(
      const Teuchos::Array<std::string> &elementBlocks,
      const Omega_h::MeshDimSets &       elementsets);

  typename Fields::index_array_type &getUserConductivityElementIDs();
};

template <
    ConductivityModelType ConductivityType,
    int                   SpatialDim>
struct ConductivityModel
    : public ConductivityModelBase<SpatialDim> {
  typedef lgr::Fields<SpatialDim> Fields;
  Scalar Conductivity(int ielem, int state) const = 0;
  void initializeElements(const Fields &mesh_fields) override { return; }
  void updateElements(const Fields &mesh_fields, int state) override { return; }
};

//curiously recurring template pattern
template <
    class Derived,
    int SpatialDim>
struct CRTP_ConductivityModelBase
    : public ConductivityModelBase<SpatialDim> {
  typedef lgr::Fields<SpatialDim> Fields;
  typedef
      typename lgr::ConductivityModelBase<SpatialDim>
          ConductivityModelBase;

  void initializeElements(const Fields &mesh_fields) override;
  void updateElements(const Fields &mesh_fields, int state) override;
};

//partial template specialization: Conductivity model 0: constant
template <int SpatialDim>
struct ConductivityModel<
    ConductivityModelType::CONSTANT,
    SpatialDim>
    : public CRTP_ConductivityModelBase<
          ConductivityModel<
              ConductivityModelType::CONSTANT,
              SpatialDim>,
          SpatialDim> {
  typedef lgr::Fields<SpatialDim> Fields;

  const Scalar conductivity_;

  ConductivityModel(
      int                           arg_user_id,
      Fields &                      meshFields,
      const Teuchos::ParameterList &matData);

  KOKKOS_INLINE_FUNCTION
  Scalar Conductivity(int /*ielem*/, int /*state*/) const { return conductivity_; }
};


template <int SpatialDim>
std::shared_ptr<
    ConductivityModelBase<SpatialDim>>
createConductivityModel(
    int                         arg_user_Conductivity_id,
    const ConductivityModelType matModelType,
    lgr::Fields<SpatialDim> &mesh_fields,
    const Teuchos::ParameterList &                      matData);

template <int SpatialDim>
void createConductivityModels(
    const Teuchos::ParameterList &ConductivityModelParameterList,
    lgr::Fields<SpatialDim> &mesh_fields,
    const Omega_h::MeshDimSets &                        elementSets,
    std::list<std::shared_ptr<
        ConductivityModelBase<SpatialDim>>>
        &theConductivityModels);

#define LGR_EXPL_INST_DECL(SpatialDim) \
extern template \
std::shared_ptr<ConductivityModelBase<SpatialDim>> \
createConductivityModel( \
    int                         arg_user_Conductivity_id, \
    const ConductivityModelType matModelType, \
    lgr::Fields<SpatialDim> &mesh_fields, \
    const Teuchos::ParameterList &                      matData); \
extern template \
void createConductivityModels( \
    const Teuchos::ParameterList &ConductivityModelParameterList, \
    lgr::Fields<SpatialDim> &mesh_fields, \
    const Omega_h::MeshDimSets &                        elementSets, \
    std::list<std::shared_ptr<ConductivityModelBase<SpatialDim>>> \
        &theConductivityModels);
LGR_EXPL_INST_DECL(3)
LGR_EXPL_INST_DECL(2)
#undef LGR_EXPL_INST_DECL

}  //end namespace lgr

#endif  //CONDUCTIVITY_MODELS_HPP
