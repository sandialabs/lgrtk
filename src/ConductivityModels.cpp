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

#include "ConductivityModels.hpp"

#include <FieldDB.hpp>
#include <Fields.hpp>
#include <Kokkos_Core.hpp>
#include <Teuchos_ParameterList.hpp>
#include <cmath>

#include "LGRLambda.hpp"
#include "ErrorHandling.hpp"
#include "MeshIO.hpp"

#include "Omega_h_eigen.hpp"
#include "Omega_h_matrix.hpp"
#include "FieldsEnum.hpp"

#include "TensorOperations_inline.hpp"

namespace lgr {

ConductivityModelType ConductivityInputSelector(
    const std::string &ConductivityModelString) {
  const std::string     ConductivityModelChoices[] = {"Constant"
                                                     };
  ConductivityModelType result =
      ConductivityModelType::NUMBER_OF_CONDUCTIVITY_MODELS;
  for (int i = 0; i < static_cast<int>(
                          ConductivityModelType::NUMBER_OF_CONDUCTIVITY_MODELS);
       ++i) {
    if (ConductivityModelString == ConductivityModelChoices[i]) {
      result = static_cast<ConductivityModelType>(i);
      break;
    }
  }
  return result;
}

std::string ConductivityInputSelector(const ConductivityModelType type) {
  const std::string ConductivityModelChoices[] = {"Constant"
                                                  };
  const int         i = static_cast<int>(type);
  return ConductivityModelChoices[i];
}

template <int spaceDim>
ConductivityModelBase<spaceDim>::ConductivityModelBase() : user_id_(-1) {}
template <int spaceDim>
ConductivityModelBase<spaceDim>::ConductivityModelBase(int arg_user_id) : user_id_(arg_user_id) {}

template <int spaceDim>
void ConductivityModelBase<spaceDim>::createElementIDs(
    const Teuchos::Array<std::string> &elementBlocks,
    const Omega_h::MeshDimSets &       elementsets) {
  //count total number of elements
  int n = 0;
  for (const std::string &blockname : elementBlocks) {
    auto esIter = elementsets.find(blockname);
    auto elementLids = (esIter->second);
    n += elementLids.size();
  }

  std::string name = "User conductivity element ids for model ";
  name += std::to_string(user_id_);
  userConductivityElementIDs_ = typename Fields::index_array_type(name, n);
  auto local_ids = userConductivityElementIDs_;

  //fill in array of element numbers
  int offset = 0;
  for (const std::string &blockname : elementBlocks) {
    auto esIter = elementsets.find(blockname);
    LGR_THROW_IF(
        esIter == elementsets.end(),
        " ERROR:Block name not found:" << blockname);
    auto elementLids = (esIter->second);
    auto f = LAMBDA_EXPRESSION(int i) {
      local_ids(offset + i) = elementLids[i];
    };
    Kokkos::parallel_for(elementLids.size(), f);
    offset += elementLids.size();
  }
}

template <int spaceDim>
typename ConductivityModelBase<spaceDim>::Fields::index_array_type& 
ConductivityModelBase<spaceDim>::getUserConductivityElementIDs() {
  return userConductivityElementIDs_;
}

template <class Derived, int SpatialDim>
void CRTP_ConductivityModelBase<Derived, SpatialDim>::initializeElements(const Fields &mesh_fields) {
  this->updateElements(mesh_fields, 0);
  this->updateElements(mesh_fields, 1);
}

template <class Derived, int SpatialDim>
void CRTP_ConductivityModelBase<Derived, SpatialDim>::updateElements(
    const Fields&, int state) {
  const Derived derivedThis = *(static_cast<Derived *>(this));
  typename Fields::index_array_type &elementID =
      this->getUserConductivityElementIDs();
  const typename Fields::array_type &conductivity = Conductivity<Fields>();

  auto updateEL = LAMBDA_EXPRESSION(int linearIndex) {
    //update Conductivity model
    const int    ielem = elementID(linearIndex);
    const Scalar c = derivedThis.Conductivity(ielem, state);
    conductivity(ielem) = c;
  };  //end lambda updateEL

  Kokkos::parallel_for(elementID.extent(0), updateEL);
}

template <int SpatialDim>
ConductivityModel<ConductivityModelType::CONSTANT, SpatialDim>::
ConductivityModel(
      int                           arg_user_id,
      Fields&,
      const Teuchos::ParameterList &matData)
      : conductivity_(matData.get<double>("conductivity")) {
  ConductivityModelBase<SpatialDim>::setUserID_(
      arg_user_id);
}


template <int SpatialDim>
std::shared_ptr<ConductivityModelBase<SpatialDim>>
createConductivityModel(
    int                         arg_user_Conductivity_id,
    const ConductivityModelType matModelType,
    lgr::Fields<SpatialDim> &mesh_fields,
    const Teuchos::ParameterList &                      matData) {
  std::shared_ptr<ConductivityModelBase<SpatialDim>>
      theConductivityModel(0);
  switch (matModelType) {
    case ConductivityModelType::CONSTANT:
      theConductivityModel = std::make_shared<ConductivityModel<
          ConductivityModelType::CONSTANT,
          SpatialDim>>(arg_user_Conductivity_id, mesh_fields, matData);
      break;
    default:
      break;
  }
  return theConductivityModel;
}

template <int SpatialDim>
void createConductivityModels(
    const Teuchos::ParameterList &ConductivityModelParameterList,
    lgr::Fields<SpatialDim> &mesh_fields,
    const Omega_h::MeshDimSets &                        elementSets,
    std::list<std::shared_ptr<
        ConductivityModelBase<SpatialDim>>>
        &theConductivityModels) {
  for (Teuchos::ParameterList::ConstIterator i =
           ConductivityModelParameterList.begin();
       i != ConductivityModelParameterList.end(); ++i) {
    if (ConductivityModelParameterList.entry(i).isList()) {
      const std::string &name_i = ConductivityModelParameterList.name(i);

      const Teuchos::ParameterList &sublist =
          ConductivityModelParameterList.sublist(name_i);

      const int         arg_user_Conductivity_id = sublist.get<int>("user id");
      const std::string ConductivityModelString =
          sublist.get<std::string>("Model Type");

      const ConductivityModelType ConductivityType =
          ConductivityInputSelector(ConductivityModelString);

      std::shared_ptr<
          ConductivityModelBase<SpatialDim>>
          ConductivityModel = createConductivityModel(
              arg_user_Conductivity_id, ConductivityType, mesh_fields, sublist);

      Teuchos::Array<std::string>    elementBlocks;
      const Teuchos::ParameterEntry &pe = sublist.getEntry("Element Block");
      if (pe.isArray()) {
        elementBlocks =
            sublist.get<Teuchos::Array<std::string>>("Element Block");
      } else {
        const std::string &elemblock =
            sublist.get<std::string>("Element Block");
        elementBlocks.push_back(elemblock);
      }
      ConductivityModel->createElementIDs(elementBlocks, elementSets);

      theConductivityModels.push_back(ConductivityModel);
    }  //end   if ( entry_i.isList() )
  }    //end  for ( Teuchos::ParameterList::ConstIterator i
}

#define LGR_EXPL_INST(SpatialDim) \
template \
std::shared_ptr<ConductivityModelBase<SpatialDim>> \
createConductivityModel( \
    int                         arg_user_Conductivity_id, \
    const ConductivityModelType matModelType, \
    lgr::Fields<SpatialDim> &mesh_fields, \
    const Teuchos::ParameterList &                      matData); \
template \
void createConductivityModels( \
    const Teuchos::ParameterList &ConductivityModelParameterList, \
    lgr::Fields<SpatialDim> &mesh_fields, \
    const Omega_h::MeshDimSets &                        elementSets, \
    std::list<std::shared_ptr<ConductivityModelBase<SpatialDim>>> \
        &theConductivityModels);
LGR_EXPL_INST(3)
LGR_EXPL_INST(2)
#undef LGR_EXPL_INST

}  //end namespace lgr
