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

#include "MaterialModels_inline.hpp"
#include "ErrorHandling.hpp"

namespace lgr {

MaterialModelType materialInputSelector(
    const std::string &materialModelString) {
  const std::string materialModelChoices[] = {
      "neo hookean", "ideal gas", "mie gruneisen", "j2 plasticity"};
  MaterialModelType result = MaterialModelType::NUMBER_OF_MATERIAL_MODELS;
  for (int i = 0;
       i < static_cast<int>(MaterialModelType::NUMBER_OF_MATERIAL_MODELS);
       ++i) {
    if (materialModelString == materialModelChoices[i]) {
      result = static_cast<MaterialModelType>(i);
      break;
    }
  }
  return result;
}

std::string materialInputSelector(const MaterialModelType type) {
  const std::string materialModelChoices[] = {
      "neo hookean", "ideal gas", "mie gruneisen", "j2 plasticity"};
  const int i = static_cast<int>(type);
  return materialModelChoices[i];
}

template <int SpatialDim>
void MaterialModelBase<SpatialDim>::setUserID_(int arg_user_id) { user_id_ = arg_user_id; }

template <int SpatialDim>
MaterialModelBase<SpatialDim>::MaterialModelBase() : user_id_(-1) {}

template <int SpatialDim>
MaterialModelBase<SpatialDim>::MaterialModelBase(int arg_user_id) : user_id_(arg_user_id) {}

template <int SpatialDim>
void MaterialModelBase<SpatialDim>::createElementIDs(
    const Teuchos::Array<std::string> &elementBlocks,
    const Omega_h::MeshDimSets &       elementsets) {
  //count total number of elements
  int n = 0;
  for (const std::string &blockname : elementBlocks) {
    auto esIter = elementsets.find(blockname);
    auto elementLids = (esIter->second);
    n += elementLids.size();
  }

  std::string name = "user material element ids for material ";
  name += std::to_string(user_id_);
  userMaterialElementIDs_ = typename Fields::index_array_type(name, n);
  auto local_ids = userMaterialElementIDs_;

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

template <int SpatialDim>
typename MaterialModelBase<SpatialDim>::Fields::index_array_type &
MaterialModelBase<SpatialDim>::getUserMaterialElementIDs() {
  return userMaterialElementIDs_;
}

template <
    MaterialModelType materialType,
    int               SpatialDim>
void MaterialModel<materialType, SpatialDim>::initializeElements(const Fields &) { }

template <
    MaterialModelType materialType,
    int               SpatialDim>
void MaterialModel<materialType, SpatialDim>::updateElements(const Fields &, int, const double, const double) { }

template <int SpatialDim>
std::shared_ptr<MaterialModelBase<SpatialDim>>
createMaterialModel(
    int                                                 arg_user_material_id,
    const MaterialModelType                             matModelType,
    lgr::Fields<SpatialDim> &mesh_fields,
    const Teuchos::ParameterList &                      matData) {
  std::shared_ptr<MaterialModelBase<SpatialDim>>
      theMaterialModel(0);
  switch (matModelType) {
    case MaterialModelType::NEO_HOOKEAN:
      theMaterialModel = std::make_shared<MaterialModel<
          MaterialModelType::NEO_HOOKEAN, SpatialDim>>(
          arg_user_material_id, mesh_fields, matData);
      break;
    case MaterialModelType::IDEAL_GAS:
      theMaterialModel = std::make_shared<MaterialModel<
          MaterialModelType::IDEAL_GAS, SpatialDim>>(
          arg_user_material_id, mesh_fields, matData);
      break;
    case MaterialModelType::MIE_GRUNEISEN:
      theMaterialModel = std::make_shared<MaterialModel<
          MaterialModelType::MIE_GRUNEISEN,
          SpatialDim>>(arg_user_material_id, mesh_fields, matData);
      break;
    case MaterialModelType::J2_PLASTICITY:
      theMaterialModel = std::make_shared<MaterialModel<
          MaterialModelType::J2_PLASTICITY,
          SpatialDim>>(arg_user_material_id, mesh_fields, matData);
      break;
    default:
      break;
  }
  return theMaterialModel;
}

template <int SpatialDim>
void createMaterialModels(
    const Teuchos::ParameterList &materialModelParameterList,
    lgr::Fields<SpatialDim> &mesh_fields,
    const Omega_h::MeshDimSets &                        elementSets,
    std::list<
        std::shared_ptr<MaterialModelBase<SpatialDim>>>
        &theMaterialModels) {
  for (Teuchos::ParameterList::ConstIterator i =
           materialModelParameterList.begin();
       i != materialModelParameterList.end(); ++i) {
    const Teuchos::ParameterEntry &entry_i =
        materialModelParameterList.entry(i);
    const std::string &name_i = materialModelParameterList.name(i);
    if (entry_i.isList()) {
      const Teuchos::ParameterList &sublist =
          materialModelParameterList.sublist(name_i);

      const int         arg_user_material_id = sublist.get<int>("user id");
      const std::string materialModelString =
          sublist.get<std::string>("Model Type");

      const MaterialModelType materialType =
          materialInputSelector(materialModelString);

      std::shared_ptr<MaterialModelBase<SpatialDim>>
          materialModel = createMaterialModel(
              arg_user_material_id, materialType, mesh_fields, sublist);

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
      materialModel->createElementIDs(elementBlocks, elementSets);

      theMaterialModels.push_back(materialModel);
    }  //end   if ( entry_i.isList() )
  }    //end  for ( Teuchos::ParameterList::ConstIterator i
}

#define LGR_EXPL_INST(SpatialDim) \
template class MaterialModelBase<SpatialDim>; \
template std::shared_ptr<MaterialModelBase<SpatialDim>> \
createMaterialModel( \
    int                                                 arg_user_material_id, \
    const MaterialModelType                             matModelType, \
    lgr::Fields<SpatialDim> &mesh_fields, \
    const Teuchos::ParameterList &                      matData); \
template void createMaterialModels( \
    const Teuchos::ParameterList &materialModelParameterList, \
    lgr::Fields<SpatialDim> &mesh_fields, \
    const Omega_h::MeshDimSets &                        elementSets, \
    std::list< \
        std::shared_ptr<MaterialModelBase<SpatialDim>>> \
        &theMaterialModels);
LGR_EXPL_INST(3)
LGR_EXPL_INST(2)
#undef LGR_EXPL_INST

}  //end namespace lgr
