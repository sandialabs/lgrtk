#include "FieldDB.hpp"
#include "Fields.hpp"
#include "MaterialModels.hpp"

namespace lgr {

template <int SpatialDim>
void FieldDB_Finalize() {
  typedef lgr::Fields<SpatialDim> Fields;
  FieldDB<typename Fields::array_type>::Self().clear();
  FieldDB<typename Fields::geom_array_type>::Self().clear();
  FieldDB<typename Fields::state_array_type>::Self().clear();
  FieldDB<typename Fields::elem_vector_type>::Self().clear();
  FieldDB<typename Fields::elem_tensor_type>::Self().clear();
  FieldDB<typename Fields::elem_node_geom_type>::Self().clear();
  FieldDB<typename Fields::geom_state_array_type>::Self().clear();
  FieldDB<typename Fields::elem_vector_state_type>::Self().clear();
  FieldDB<typename Fields::elem_sym_tensor_state_type>::Self().clear();
}

template void FieldDB_Finalize<1>();
template void FieldDB_Finalize<2>();
template void FieldDB_Finalize<3>();

}
