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

// because of vagaries of C++ compilers, it is important to include Omega_h_expr.hpp
// prior to things that include Teuchos_any.hpp (which InitialConditions.hpp does, albeit indirectly)
#include <utility>

#include <Omega_h_expr.hpp>

#include "ErrorHandling.hpp"
#include "InitialConditions.hpp"
#include "LGRLambda.hpp"

namespace lgr {

template <class Field>
InitialCondition<Field>::~InitialCondition(){};

template <class Field>
InitialCondition<Field>::InitialCondition(
    const std::string &n, Teuchos::ParameterList &params)
    : name(n) {
  const Teuchos::ParameterEntry *eb = params.getEntryPtr("Element Block");
  if (nullptr != eb) {
    if (eb->isArray()) {
      const Teuchos::Array<std::string> &arrayData =
          params.get<Teuchos::Array<std::string>>("Element Block");
      element_blocks.resize(arrayData.size());
      std::copy(arrayData.begin(), arrayData.end(), element_blocks.begin());
    } else
      element_blocks.push_back(params.get<std::string>("Element Block"));
  }
  const Teuchos::ParameterEntry *ns = params.getEntryPtr("Nodeset");
  if (nullptr != ns) {
    if (ns->isArray()) {
      const Teuchos::Array<std::string> &arrayData =
          params.get<Teuchos::Array<std::string>>("Nodeset");
      node_sets.resize(arrayData.size());
      std::copy(arrayData.begin(), arrayData.end(), node_sets.begin());
    } else
      node_sets.push_back(params.get<std::string>("Nodeset"));
  }
  const Teuchos::ParameterEntry *fs = params.getEntryPtr("Faceset");
  if (nullptr != fs) {
    if (fs->isArray()) {
      const Teuchos::Array<std::string> &arrayData =
          params.get<Teuchos::Array<std::string>>("Faceset");
      face_sets.resize(arrayData.size());
      std::copy(arrayData.begin(), arrayData.end(), face_sets.begin());
    } else
      face_sets.push_back(params.get<std::string>("Faceset"));
  }
}

template <class Field>
ConstantInitialCondition<Field>::ConstantInitialCondition(
    const std::string &n, Teuchos::ParameterList &params)
    : InitialCondition<Field>(n, params), value() {
  const Teuchos::ParameterEntry &pe = params.getEntry("Value");
  if (pe.isArray()) {
    const Teuchos::Array<double> &arrayData =
        params.get<Teuchos::Array<double>>("Value");

    TEUCHOS_TEST_FOR_EXCEPTION(
        arrayData.size() != Field::SpaceDim, std::logic_error,
        " Initial Condition type invalid: array length not equal to spatial "
        "dimension.");
    value.resize(Field::SpaceDim);
    std::copy(arrayData.begin(), arrayData.end(), value.begin());
  } else {
    value.resize(1);
    value[0] = params.get<double>("Value");
  }
}

template <class Field>
void ConstantInitialCondition<Field>::set(
    const Omega_h::MeshDimSets &nodesets,
    const geom_array_type       field,
    const node_coords) {
  for (const std::string &setname : this->node_sets) {
    auto nsIter = nodesets.find(setname);
    TEUCHOS_TEST_FOR_EXCEPTION(
        nsIter == nodesets.end(), std::invalid_argument,
        "\"" << this->name << "\": node set " << setname
             << " doesn't exist!\n");
    auto nodeLids = (nsIter->second);

    for (size_t s = 0; s < value.size(); ++s) {
      auto scalar = value[s];
      auto f = LAMBDA_EXPRESSION(int set_node) {
        auto node = nodeLids[set_node];
        field(node, s) = scalar;
      };
      Kokkos::parallel_for(nodeLids.size(), f);
    }
  }
}

template <class Field>
void ConstantInitialCondition<Field>::set(
    const Omega_h::MeshDimSets &elementsets,
    const array_type            field,
    const node_coords,
    const elem_node_ids) {
  for (const std::string &blockname : this->element_blocks) {
    auto esIter = elementsets.find(blockname);
    TEUCHOS_TEST_FOR_EXCEPTION(
        esIter == elementsets.end(), std::invalid_argument,
        "\"" << this->name << "\": element block " << blockname
             << " doesn't exist!\n");
    TEUCHOS_TEST_FOR_EXCEPTION(
        value.size() != 1, std::invalid_argument,
        "value for constant initial condition needs to be a scalar");
    auto elementLids = (esIter->second);
    auto scalar = value[0];
    auto f = LAMBDA_EXPRESSION(int set_elem) {
      auto elem = elementLids[set_elem];
      field(elem) = scalar;
    };
    Kokkos::parallel_for(elementLids.size(), f);
  }
}

template <class Field>
void ConstantInitialCondition<Field>::set(
    const Omega_h::MeshDimSets &,
    const Omega_h::MeshDimSets &,
    const array_type            ,
    const node_coords,
    const face_node_ids,
    const elem_face_ids                  ) {
  TEUCHOS_TEST_FOR_EXCEPTION(
    true, std::logic_error,"ConstantInitialCondition for faces not implimented.");
}
             

template <class Field>
FunctionInitialCondition<Field>::FunctionInitialCondition(
    const std::string &n, Teuchos::ParameterList &params)
    : InitialCondition<Field>(n, params)
    , expr_string(params.get<std::string>("Value")) {}

template <class Field>
void FunctionInitialCondition<Field>::set(
    const Omega_h::MeshDimSets &nodesets,
    const geom_array_type       field,
    const node_coords           coords) {
  for (const std::string &setname : this->node_sets) {
    auto nsIter = nodesets.find(setname);
    TEUCHOS_TEST_FOR_EXCEPTION(
        nsIter == nodesets.end(), std::invalid_argument,
        "\"" << this->name << "\": node set " << setname
             << " doesn't exist!\n");
    auto                   nodeLids = (nsIter->second);
    auto                   nset_nodes = nodeLids.size();
    Omega_h::Write<double> x_w(nset_nodes);
    Omega_h::Write<double> y_w(nset_nodes);
    Omega_h::Write<double> z_w(nset_nodes);
    auto prepare = LAMBDA_EXPRESSION(int set_node) {
      auto node = nodeLids[set_node];
      x_w[set_node] = coords(node, 0);
      y_w[set_node] = coords(node, 1);
      z_w[set_node] = coords(node, 2);
    };
    Kokkos::parallel_for(nset_nodes, prepare);
    Omega_h::ExprReader reader(nset_nodes, Field::SpaceDim);
    reader.register_variable("x", Teuchos::any(Omega_h::Reals(x_w)));
    reader.register_variable("y", Teuchos::any(Omega_h::Reals(y_w)));
    reader.register_variable("z", Teuchos::any(Omega_h::Reals(z_w)));
    Teuchos::any result;
    reader.read_string(result, this->expr_string, this->name);
    reader.repeat(result);
    auto field_osh = Teuchos::any_cast<Omega_h::Reals>(result);
    auto save = LAMBDA_EXPRESSION(int set_node) {
      auto node = nodeLids[set_node];
      for (int s = 0; s < Field::SpaceDim; ++s)
        field(node, s) = field_osh[set_node * Field::SpaceDim + s];
    };
    Kokkos::parallel_for(nset_nodes, save);
  }
  return;
}

template <class Field>
void FunctionInitialCondition<Field>::set(
    const Omega_h::MeshDimSets &elementsets,
    const array_type            field,
    const node_coords           coords,
    const elem_node_ids         elem_node_id) {
  for (const std::string &blockname : this->element_blocks) {
    auto esIter = elementsets.find(blockname);
    TEUCHOS_TEST_FOR_EXCEPTION(
        esIter == elementsets.end(), std::invalid_argument,
        "\"" << this->name << "\": element block " << blockname
             << " doesn't exist!\n");
    auto                   elementLids = (esIter->second);
    auto                   nset_elems = elementLids.size();
    Omega_h::Write<double> x_w(nset_elems);
    Omega_h::Write<double> y_w(nset_elems);
    Omega_h::Write<double> z_w(nset_elems);
    const elem_node_ids    node_id = elem_node_id;
    auto prepare = LAMBDA_EXPRESSION(int set_elem) {
      x_w[set_elem] = 0;
      y_w[set_elem] = 0;
      z_w[set_elem] = 0;
      auto elem = elementLids[set_elem];
      for (int i = 0; i < Field::ElemNodeCount; ++i) {
        auto node = node_id(elem, i);
        x_w[set_elem] += coords(node, 0);
        y_w[set_elem] += coords(node, 1);
        z_w[set_elem] += coords(node, 2);
      }
      x_w[set_elem] /= Field::ElemNodeCount;
      y_w[set_elem] /= Field::ElemNodeCount;
      z_w[set_elem] /= Field::ElemNodeCount;
    };
    Kokkos::parallel_for(nset_elems, prepare);
    Omega_h::ExprReader reader(nset_elems, Field::SpaceDim);
    reader.register_variable("x", Teuchos::any(Omega_h::Reals(x_w)));
    reader.register_variable("y", Teuchos::any(Omega_h::Reals(y_w)));
    reader.register_variable("z", Teuchos::any(Omega_h::Reals(z_w)));
    Teuchos::any result;
    reader.read_string(result, this->expr_string, this->name);
    reader.repeat(result);
    auto field_osh = Teuchos::any_cast<Omega_h::Reals>(result);
    auto save = LAMBDA_EXPRESSION(int set_elem) {
      auto elem = elementLids[set_elem];
      field(elem) = field_osh[set_elem];
    };
    Kokkos::parallel_for(nset_elems, save);
  }
}

template <class Field>
void FunctionInitialCondition<Field>::set(
    const Omega_h::MeshDimSets &facesets,
    const Omega_h::MeshDimSets &elemsets,
    const array_type            field,
    const node_coords           coords,
    const face_node_ids         face_node_id,
    const elem_face_ids         elem_face_id) {

  constexpr int N = Field::FaceNodeCount;
  constexpr int F = Field::ElemFaceCount;
  constexpr int D = Field::SpaceDim;
  for (const std::string &faceset : this->face_sets) {
    auto fsIter = facesets.find(faceset);
    TEUCHOS_TEST_FOR_EXCEPTION(
        fsIter == facesets.end(), std::invalid_argument,
        "\"" << this->name << "\": faceset block " << faceset  
             << " doesn't exist!\n");
    const auto     faceLids = (fsIter->second);
    const auto   nset_faces = faceLids.size();
    // Edge Centers
    Omega_h::Write<double> Cx(N*nset_faces);
    Omega_h::Write<double> Cy(N*nset_faces);
    Omega_h::Write<double> Cz(N*nset_faces);
    // Edge Vectors
    Omega_h::Write<double> Vx(N*nset_faces);
    Omega_h::Write<double> Vy(N*nset_faces);
    Omega_h::Write<double> Vz(N*nset_faces);
    
    const node_coords   cord = coords;
    const face_node_ids ids  = face_node_id;
    auto prepare = LAMBDA_EXPRESSION(int set_face) {
      const auto face = faceLids[set_face];
      int nodes[N];              
      for (int i = 0; i < N; ++i) 
        nodes[i] = ids(face, i);
      double X[N][3];
      for (int i = 0; i < N; ++i) 
        for (int j = 0; j < D; ++j) 
           X[i][j] = cord(nodes[i], j);
      
      const auto offset = set_face*N;
      for (int i = 0; i < N; ++i) {
        const int j = (i+1)%N;
        if (0<D) Vx[offset+i] =  X[j][0] - X[i][0];
        if (1<D) Vy[offset+i] =  X[j][1] - X[i][1];
        if (2<D) Vz[offset+i] =  X[j][2] - X[i][2];
        if (0<D) Cx[offset+i] = (X[j][0] + X[i][0])/2;
        if (1<D) Cy[offset+i] = (X[j][1] + X[i][1])/2;
        if (2<D) Cz[offset+i] = (X[j][2] + X[i][2])/2;
      }
    };
    Kokkos::parallel_for(nset_faces, prepare);
    const std::vector<std::string> S={"x","y","z"};
    Omega_h::ExprReader reader(N*nset_faces, D);
    reader.register_variable(S[0], Teuchos::any(Omega_h::Reals(Cx)));
    reader.register_variable(S[1], Teuchos::any(Omega_h::Reals(Cy)));
    reader.register_variable(S[2], Teuchos::any(Omega_h::Reals(Cz)));
    Teuchos::any result;
    reader.read_string(result, this->expr_string, this->name);
    reader.repeat(result);
    Omega_h::Reals field_osh = Teuchos::any_cast<Omega_h::Reals>(result);

    const array_type fld=field;
    auto save = LAMBDA_EXPRESSION(int set_face) {
      double Fl[N][3];
      double v [N][3];
      const auto face = faceLids[set_face];
      const auto offset = set_face*N;
      for (int i = 0; i < N; ++i) 
        for (int s = 0; s < D; ++s)
          Fl[i][s] = field_osh[(offset+i) * D + s];
      for (int i = 0; i < N; ++i) {
        if (0<D) v[i][0] = Vx[offset+i];
        if (1<D) v[i][1] = Vy[offset+i];
        if (2<D) v[i][2] = Vz[offset+i];
      }
      double flux=0;
      for (int i = 0; i < N; ++i) 
        for (int j = 0; j < D; ++j) 
          flux += Fl[i][j]*v[i][j];

      fld(face) = flux;
    };
    Kokkos::parallel_for(nset_faces, save);
  }

  for (const std::string &blockname : this->element_blocks) {
    auto esIter = elemsets.find(blockname);
    TEUCHOS_TEST_FOR_EXCEPTION(
        esIter == elemsets.end(), std::invalid_argument,
        "\"" << this->name << "\": std block " << blockname
             << " doesn't exist!\n");
    const auto elementLids = (esIter->second);
    const int   nset_elems = elementLids.size();
    const int  nset_faces = F*nset_elems;
    // Edge Centers
    Omega_h::Write<double> Cx(N*nset_faces);
    Omega_h::Write<double> Cy(N*nset_faces);
    Omega_h::Write<double> Cz(N*nset_faces);
    // Edge Vectors
    Omega_h::Write<double> Vx(N*nset_faces);
    Omega_h::Write<double> Vy(N*nset_faces);
    Omega_h::Write<double> Vz(N*nset_faces);
    
    const node_coords   cord = coords;
    const face_node_ids ids  = face_node_id;
    const elem_face_ids fids = elem_face_id;
    auto prepare = LAMBDA_EXPRESSION(int set_face) {
      const int  el = elementLids[set_face/F];
      const int   f = set_face%F;
      const auto face = fids(el,f);
      int nodes[N];              
      for (int i = 0; i < N; ++i) 
        nodes[i] = ids(face, i);
      double X[N][3];
      for (int i = 0; i < N; ++i) 
        for (int j = 0; j < D; ++j) 
           X[i][j] = cord(nodes[i], j);
      
      const auto offset = set_face*N;
      for (int i = 0; i < N; ++i) {
        const int j = (i+1)%N;
        if (0<D) Vx[offset+i] =  X[j][0] - X[i][0];
        if (1<D) Vy[offset+i] =  X[j][1] - X[i][1];
        if (2<D) Vz[offset+i] =  X[j][2] - X[i][2];
        if (0<D) Cx[offset+i] = (X[j][0] + X[i][0])/2;
        if (1<D) Cy[offset+i] = (X[j][1] + X[i][1])/2;
        if (2<D) Cz[offset+i] = (X[j][2] + X[i][2])/2;
      }
    };
    Kokkos::parallel_for(nset_faces, prepare);
    const std::vector<std::string> S={"x","y","z"};
    Omega_h::ExprReader reader(N*nset_faces, D);
    reader.register_variable(S[0], Teuchos::any(Omega_h::Reals(Cx)));
    reader.register_variable(S[1], Teuchos::any(Omega_h::Reals(Cy)));
    reader.register_variable(S[2], Teuchos::any(Omega_h::Reals(Cz)));
    Teuchos::any result;
    reader.read_string(result, this->expr_string, this->name);
    reader.repeat(result);
    Omega_h::Reals field_osh = Teuchos::any_cast<Omega_h::Reals>(result);

    const array_type fld=field;
    auto save = LAMBDA_EXPRESSION(int set_face) {
      double Fl[N][3];
      double v [N][3];
      const int  el = elementLids[set_face/Field::ElemFaceCount];
      const int   f = set_face%Field::ElemFaceCount;
      const auto face = fids(el,f);
      const auto offset = set_face*N;
      for (int i = 0; i < N; ++i) 
        for (int s = 0; s < D; ++s)
          Fl[i][s] = field_osh[(offset+i) * D + s];
      for (int i = 0; i < N; ++i) {
        if (0<D) v[i][0] = Vx[offset+i];
        if (1<D) v[i][1] = Vy[offset+i];
        if (2<D) v[i][2] = Vz[offset+i];
      }
      double flux=0;
      for (int i = 0; i < N; ++i) 
        for (int j = 0; j < D; ++j) 
          flux += Fl[i][j]*v[i][j];

      fld(face) = flux;
    };
    Kokkos::parallel_for(nset_faces, save);
  }
}

template <class Field>
InitialConditions<Field>::InitialConditions(Teuchos::ParameterList &params)
    : velocity_initial_conditions()
    , displacement_initial_conditions()
    , density_initial_conditions()
    , internal_energy_initial_conditions()
    , face_flux_initial_conditions() {
  for (Teuchos::ParameterList::ConstIterator i = params.begin();
       i != params.end(); ++i) {
    const Teuchos::ParameterEntry &entry_i = params.entry(i);
    const std::string &            name_i = params.name(i);
    if (entry_i.isList()) {
      Teuchos::ParameterList &sublist = params.sublist(name_i);
      const std::string       type = sublist.get<std::string>("Type");
      const std::string       var = sublist.get<std::string>("Variable");
      std::shared_ptr<InitialCondition<Field>> ic;
      if ("Constant" == type)
        ic.reset(new ConstantInitialCondition<Field>(name_i, sublist));
      else if ("Function" == type)
        ic.reset(new FunctionInitialCondition<Field>(name_i, sublist));
      else
        TEUCHOS_TEST_FOR_EXCEPTION(
            true, std::logic_error,
            " Initial Condition type invalid: Not Constant or Function.");
      if (var == "Velocity")
        velocity_initial_conditions.push_back(ic);
      else if (var == "Displacement")
        displacement_initial_conditions.push_back(ic);
      else if (var == "Density")
        density_initial_conditions.push_back(ic);
      else if (var == "Specific Internal Energy")
        internal_energy_initial_conditions.push_back(ic);
      else if (var == "Magnetic Flux Vector Potential") {
        TEUCHOS_TEST_FOR_EXCEPTION(
          "Constant" == type, std::logic_error,
          " Initial Condition type invalid: Constant can not be used for Magnetic Flux.");
        if (Field::SpaceDim != 3) { // Quite "statement is unreachable" warning with cuda compiler
          TEUCHOS_TEST_FOR_EXCEPTION(
            Field::SpaceDim != 3, std::logic_error,
            " Initial Condition type invalid: Magnetic Flux only valid in three dimensions.");
        } else {
          face_flux_initial_conditions.push_back(ic);
        }
      }
      else if (var == "Internal Energy") {
        std::cout << "WARNING: 'Internal Energy' initial condition specifier "
                     "is deprecated.  Please use 'Specific Internal Energy' "
                     "instead.\n";
        internal_energy_initial_conditions.push_back(ic);
      } else
        std::cout << " Parameter in Initial Conditions block not valid.  Field "
                     "name invalid."
                  << std::endl;
    }
  }
}

template <class Field>
void InitialConditions<Field>::set(
    const Omega_h::MeshDimSets &nodesets,
    const geom_state_array_type velocity_state,
    const geom_array_type       displacement,
    const FEMesh &              femesh) {
  if (!velocity_initial_conditions.empty()) {
    for (int i = 0; i < Field::NumStates; ++i) {
      geom_array_type velocity(Field::getGeomFromSA(velocity_state, i));
      for (auto &ic : velocity_initial_conditions) {
        ic->set(nodesets, velocity, femesh.node_coords);
      }
    }
  }
  if (!displacement_initial_conditions.empty()) {
    for (auto &ic : displacement_initial_conditions) {
      ic->set(nodesets, displacement, femesh.node_coords);
    }
  }
}

template <class Field>
void InitialConditions<Field>::set(
    const Omega_h::MeshDimSets &elementsets,
    const state_array_type      density_state,
    const state_array_type      internal_energy_per_unit_mass_state,
    const FEMesh &              femesh) {
  if (!density_initial_conditions.empty()) {
    for (int i = 0; i < Field::NumStates; ++i) {
      array_type density(Field::getFromSA(density_state, i));
      for (auto &ic : density_initial_conditions) {
        ic->set(elementsets, density, femesh.node_coords, femesh.elem_node_ids);
      }
    }
  }
  if (!internal_energy_initial_conditions.empty()) {
    for (int i = 0; i < Field::NumStates; ++i) {
      array_type internal_energy_per_unit_mass(
          Field::getFromSA(internal_energy_per_unit_mass_state, i));
      for (auto &ic : internal_energy_initial_conditions) {
        ic->set(
            elementsets, internal_energy_per_unit_mass, femesh.node_coords,
            femesh.elem_node_ids);
      }
    }
  }
}

template <class Field>
void InitialConditions<Field>::set(
    const Omega_h::MeshDimSets &facesets,
    const Omega_h::MeshDimSets &elemsets,
    const array_type            magnetic_face_flux,
    const FEMesh &              femesh) {
  for (auto &ic : face_flux_initial_conditions) {
    ic->set(facesets, elemsets, 
      magnetic_face_flux, femesh.node_coords, 
      femesh.face_node_ids,
      femesh.elem_face_ids);
  }
}

#define LGR_EXPL_INST(SpatialDim) \
template class InitialCondition<Fields<SpatialDim>>; \
template class ConstantInitialCondition<Fields<SpatialDim>>; \
template class FunctionInitialCondition<Fields<SpatialDim>>; \
template class InitialConditions<Fields<SpatialDim>>;
LGR_EXPL_INST(3)
LGR_EXPL_INST(2)
#undef LGR_EXPL_INST

} /* namespace lgr */
