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

#include "Fields.hpp"
#include "FieldDB.hpp"
#include "LGRLambda.hpp"
#include "FieldsEnum.hpp"
#include <Omega_h_mesh.hpp>

namespace lgr {

template <int SpatialDim>
Fields<SpatialDim>::Fields(const FEMesh& mesh, Teuchos::ParameterList& data)
    : femesh(mesh), fieldData(data) {
  allocate_and_resize_fields();
}

template <int SpatialDim>
void Fields<SpatialDim>::resize() {
  femesh.resetSizes();
  femesh.reAlloc();
  femesh.updateMesh();
  allocate_and_resize_fields();
}

template <int SpatialDim>
void Fields<SpatialDim>::allocate_and_resize_fields() {
  Kokkos::realloc(
      FieldDB<array_type>::Self()["normed indicator"], femesh.nnodes);
  Kokkos::realloc(FieldDB<array_type>::Self()["nodal mass"], femesh.nnodes);
  Kokkos::realloc(FieldDB<array_type>::Self()["nodal volume"], femesh.nnodes);
  Kokkos::realloc(FieldDB<array_type>::Self()["nodal pressure"], femesh.nnodes);
  Kokkos::realloc(
      FieldDB<array_type>::Self()["nodal pressure increment"], femesh.nnodes);
  Kokkos::realloc(
      FieldDB<array_type>::Self()["nodal internal energy"], femesh.nnodes);
  Kokkos::realloc(
      FieldDB<array_type>::Self()["nodal size field"], femesh.nnodes);

  Kokkos::realloc(FieldDB<array_type>::Self()["elem volume"], femesh.nelems);
  Kokkos::realloc(
      FieldDB<array_type>::Self()["internal energy density"], femesh.nelems);
  Kokkos::realloc(FieldDB<array_type>::Self()["elem mass"], femesh.nelems);
  Kokkos::realloc(
      FieldDB<array_type>::Self()["element internal energy"], femesh.nelems);
  Kokkos::realloc(
      FieldDB<array_type>::Self()["element time step"], femesh.nelems);
  Kokkos::realloc(
      FieldDB<array_type>::Self()["fine scale pressure"], femesh.nelems);
  Kokkos::realloc(FieldDB<array_type>::Self()["user mat id"], femesh.nelems);

  Kokkos::realloc(
      FieldDB<state_array_type>::Self()["spatial deformed density"],
      femesh.nelems);
  Kokkos::realloc(
      FieldDB<state_array_type>::Self()["internal energy per unit mass"],
      femesh.nelems);
  Kokkos::realloc(
      FieldDB<state_array_type>::Self()["plane wave modulus"], femesh.nelems);
  Kokkos::realloc(
      FieldDB<state_array_type>::Self()["bulk modulus"], femesh.nelems);

  Kokkos::realloc(
      FieldDB<elem_vector_type>::Self()["fine scale velocity"], femesh.nelems);
  Kokkos::realloc(
      FieldDB<elem_vector_type>::Self()["element shock heat flux"],
      femesh.nelems);

  Kokkos::realloc(
      FieldDB<elem_tensor_type>::Self()["velocity gradient"], femesh.nelems);
  Kokkos::realloc(
      FieldDB<elem_tensor_type>::Self()["deformation gradient"], femesh.nelems);
  Kokkos::realloc(
      FieldDB<elem_tensor_type>::Self()["save the deformation gradient"],
      femesh.nelems);

  Kokkos::realloc(
      FieldDB<elem_node_geom_type>::Self()["element force"], femesh.nelems);

  Kokkos::realloc(
      FieldDB<elem_vector_state_type>::Self()["fine scale displacement"],
      femesh.nelems);

  Kokkos::realloc(
      FieldDB<array_type>::Self()["magnetic face flux"],
      femesh.nfaces);

  Kokkos::realloc(
      FieldDB<elem_sym_tensor_state_type>::Self()["stress"], femesh.nelems);

  {
    const int dimensions[] = {static_cast<int>(femesh.nnodes), SpatialDim,
                              NumStates};
    const int order[] = {1, 0, 2};
    const int rank = 3;
    const Kokkos::LayoutStride layout =
        Kokkos::LayoutStride::order_dimensions(rank, order, dimensions);
    Kokkos::realloc(FieldDB<geom_state_array_type>::Self()["velocity"], layout);
    Kokkos::realloc(
        FieldDB<geom_state_array_type>::Self()["spatial coordinates"], layout);
  }
  {
    auto layout = femesh.geom_layout;
    Kokkos::realloc(FieldDB<geom_array_type>::Self()["displacement"], layout);
    Kokkos::realloc(FieldDB<geom_array_type>::Self()["acceleration"], layout);
    Kokkos::realloc(FieldDB<geom_array_type>::Self()["internal force"], layout);
    Kokkos::realloc(
        FieldDB<geom_array_type>::Self()["nodal indicator"],
        layout);  // for error indicator
  }
  {
    const int dimensions[] = {static_cast<int>(femesh.nelems), SpatialDim};
    const int order[] = {1, 0};
    const Kokkos::LayoutStride layout =
        Kokkos::LayoutStride::order_dimensions(2, order, dimensions);
    Kokkos::realloc(
        FieldDB<geom_array_type>::Self()["element momentum"], layout);
  }
}

template <int SpatialDim>
void Fields<SpatialDim>::copyGeomToMesh(
    int                   dim,
    std::string const&    name,
    const geom_array_type from,
    bool                  should_add) const {
  auto nents = femesh.omega_h_mesh->nents(dim);
  auto out = Kokkos::View<double*>("out", nents * SpatialDim);
  auto tmp = from;
  auto f = LAMBDA_EXPRESSION(Omega_h::LO ent) {
    for (Omega_h::Int j = 0; j < SpatialDim; ++j) {
      out(ent * SpatialDim + j) = tmp(ent, j);
    }
  };
  Kokkos::parallel_for(nents, f);
  if (should_add)
    femesh.addFieldView(dim, name, SpatialDim, out);
  else
    femesh.setFieldView(dim, name, out);
}

template <int SpatialDim>
void Fields<SpatialDim>::copyGeomToMesh(
    std::string const& name, const geom_array_type from) const {
  copyGeomToMesh(0, name, from);
}

template <int SpatialDim>
void Fields<SpatialDim>::copyGeomFromMesh(
    int dim, std::string const& name, geom_array_type into) {
  femesh.copyGeomFromMesh(dim, name, into);
}

template <int SpatialDim>
void Fields<SpatialDim>::copyGeomFromMesh(
    std::string const& name, geom_array_type into) {
  femesh.copyGeomFromMesh(0, name, into);
}

template <int SpatialDim>
void Fields<SpatialDim>::copyElemScalarFromMesh(
    char const* name, array_type into) {
  auto in = femesh.getFieldView(SpatialDim, name);
  Kokkos::deep_copy(into, in);
}

template <int SpatialDim>
void Fields<SpatialDim>::copyToMesh(
    char const* name, const array_type from) const {
  /* assume all nodal fields are linearly interpolated */
  Kokkos::View<Omega_h::Real*> into("into", from.size());
  Kokkos::deep_copy(into, from);
  femesh.addFieldView(0, name, 1, into);
}

template <int SpatialDim>
void Fields<SpatialDim>::copyElemTensorToMesh(
    char const* name, const elem_tensor_type from) const {
  auto                         nelems = femesh.omega_h_mesh->nelems();
  Kokkos::View<Omega_h::Real*> into("into", nelems * TensorLength);
  assert(into.size() == from.size());
  auto f = LAMBDA_EXPRESSION(Omega_h::LO ent) {
    /* omega_h stores full tensors in column-first order.
   we assume here that K_F_XY means row X column Y. */
    for (int tensorEntryOrdinal = 0; tensorEntryOrdinal < TensorLength;
         ++tensorEntryOrdinal) {
      into(ent * TensorLength + tensorEntryOrdinal) =
          from(ent, TensorIndices<SpatialDim>(tensorEntryOrdinal));
    }
  };
  Kokkos::parallel_for(nelems, f);
  femesh.addFieldView(SpatialDim, name, TensorLength, into);
}

template <int SpatialDim>
void Fields<SpatialDim>::copyElemSymTensorToMesh(
    char const* name, const elem_sym_tensor_type from) const {
  auto                         nelems = femesh.omega_h_mesh->nelems();
  Kokkos::View<Omega_h::Real*> into("into", nelems * SymTensorLength);
  assert(into.size() == from.size());
  auto f = LAMBDA_EXPRESSION(Omega_h::LO ent) {
    for (int tensorEntryOrdinal = 0; tensorEntryOrdinal < SymTensorLength;
         ++tensorEntryOrdinal) {
      into(ent * SymTensorLength + tensorEntryOrdinal) =
          from(ent, TensorIndices<SpatialDim>(tensorEntryOrdinal));
    }
  };
  Kokkos::parallel_for(nelems, f);
  femesh.addFieldView(SpatialDim, name, SymTensorLength, into);
}


template <int SpatialDim>
void Fields<SpatialDim>::copyFromMesh(
    char const* name, array_type into) {
  auto in = femesh.getFieldView(0, name);
  Kokkos::deep_copy(into, in);
}

template <int SpatialDim>
void Fields<SpatialDim>::copyElemTensorFromMesh(
    char const* name, elem_tensor_type into) {
  auto in = femesh.getFieldView(SpatialDim, name);
  assert(in.size() == into.size());
  auto nelems = femesh.omega_h_mesh->nelems();
  auto f = LAMBDA_EXPRESSION(Omega_h::LO ent) {
    /* see copyElemTensorToMesh */
    for (int tensorEntryOrdinal = 0; tensorEntryOrdinal < TensorLength;
         ++tensorEntryOrdinal) {
      into(ent, TensorIndices<SpatialDim>(tensorEntryOrdinal)) =
          in(ent * TensorLength + tensorEntryOrdinal);
    }
  };
  Kokkos::parallel_for(nelems, f);
}

template <int SpatialDim>
void Fields<SpatialDim>::copyElemScalarToMesh(
    char const* name, const array_type from) const {
  Kokkos::View<Omega_h::Real*> into("into", from.size());
  Kokkos::deep_copy(into, from);
  femesh.addFieldView(SpatialDim, name, 1, into);
}

template <int SpatialDim>
void Fields<SpatialDim>::conformGeom(
    char const* name, geom_array_type a) {
  copyGeomToMesh(name, a);
  femesh.omega_h_mesh->sync_tag(0, name);
  copyGeomFromMesh(name, a);
  femesh.omega_h_mesh->remove_tag(0, name);
}

template <int SpatialDim>
void Fields<SpatialDim>::conform(
    char const* name, array_type a) {
  copyToMesh(name, a);
  femesh.omega_h_mesh->sync_tag(0, name);
  copyFromMesh(name, a);
  femesh.omega_h_mesh->remove_tag(0, name);
}

template <int SpatialDim>
void Fields<SpatialDim>::copyCoordsToMesh(int state) const {
  auto spcord = Coordinates<Fields>();
  auto coords = getGeomFromSA(spcord, state);
  copyGeomToMesh(0, "coordinates", coords, false);
}

template <int SpatialDim>
void Fields<SpatialDim>::copyTagsToMesh(
    Omega_h::TagSet const& tags, int state) const {
  auto mesh = femesh.omega_h_mesh;
  auto dim = size_t(SpatialDim);
  if (tags[0].count("coordinates")) {
    copyCoordsToMesh(state);
  }
  if (tags[0].count("vel")) {
    auto vel = getGeomFromSA(Velocity<Fields>(), state);
    copyGeomToMesh("vel", vel);
  }
  if (tags[0].count("velocity")) {
    auto vel = getGeomFromSA(Velocity<Fields>(), state);
    copyGeomToMesh("velocity", vel);
  }
  if (tags[0].count("force")) {
    copyGeomToMesh("force", InternalForce<Fields>());
  }
  if (tags[0].count("nodal_mass")) {
    copyToMesh("nodal_mass", NodalMass<Fields>());
  }
  if (tags[0].count("nodal_pressure")) {
    copyToMesh("nodal_pressure", FieldDB<array_type>::Self()["nodal pressure"]);
  }
  if (tags[0].count("mass")) {
    copyToMesh("mass", NodalMass<Fields>());
  }
  if (tags[0].count("potential")) {
    copyToMesh("potential", ElectricPotential<Fields>());
  }
  if (tags[dim].count("mass_density")) {
    auto dens = getFromSA(MassDensity<Fields>(), state);
    copyElemScalarToMesh("mass_density", dens);
  }
  if (tags[dim].count("spatialDensity")) {
    auto dens = getFromSA(MassDensity<Fields>(), state);
    copyElemScalarToMesh("spatialDensity", dens);
  }
  if (tags[dim].count("userMatID")) {
    copyElemScalarToMesh("userMatID", UserMatID<Fields>());
  }
  if (tags[dim].count("material")) {
    copyElemScalarToMesh("material", UserMatID<Fields>());
  }
  if (tags[dim].count("time_step")) {
    copyElemScalarToMesh("time_step", ElementTimeStep<Fields>());
  }
  if (tags[dim].count("internal_energy_density")) {
    copyElemScalarToMesh(
        "internal_energy_density", InternalEnergyDensity<Fields>());
  }
  if (tags[dim].count("internal_energy")) {
    copyElemScalarToMesh("internal_energy", ElementInternalEnergy<Fields>());
  }
  if (tags[dim].count("joule_energy")) {
    copyElemScalarToMesh("joule_energy", ElementJouleEnergy<Fields>());
  }
  if (tags[dim].count("internal_energy_per_mass")) {
    auto e_over_m = getFromSA(InternalEnergyPerUnitMass<Fields>(), state);
    copyElemScalarToMesh("internal_energy_per_mass", e_over_m);
  }
  if (tags[dim].count("mass")) {
    copyElemScalarToMesh("mass", ElementMass<Fields>());
  }
  if (tags[dim].count("conductivity")) {
    copyElemScalarToMesh("conductivity", Conductivity<Fields>());
  }
  if (tags[dim].count("deformation_gradient")) {
    copyElemTensorToMesh("deformation_gradient", DeformationGradient<Fields>());
  }
  if (tags[dim].count("stress")) {
    copyElemSymTensorToMesh("stress", getFromSymTensorSA(Stress<Fields>(),state));
  }
  if (tags[dim].count("fine_scale_displacement")) {
    copyGeomToMesh(
        SpatialDim, "fine_scale_displacement",
        getGeomFromSA(FineScaleDisplacement<Fields>(), state));
  }
  if (tags[dim].count("quality")) mesh->ask_qualities();
}

template <int SpatialDim>
void Fields<SpatialDim>::copyCoordsFromMesh(int state) {
  auto spcord = Coordinates<Fields>();
  auto coords = getGeomFromSA(spcord, state);
  copyGeomFromMesh(0, "coordinates", coords);
}

template <int SpatialDim>
void Fields<SpatialDim>::copyTagsFromMesh(
    Omega_h::TagSet const& tags, int state) {
  auto elem_dim = size_t(SpatialDim);
  auto vert_dim = size_t(0);

  // Node data
  if (tags[vert_dim].count("coordinates")) {
    copyCoordsFromMesh(state);
  }
  if (tags[vert_dim].count("vel")) {
    auto vel =
        getGeomFromSA(Velocity<Fields<SpatialDim>>(), state);
    copyGeomFromMesh("vel", vel);
  }
  if (tags[vert_dim].count("velocity")) {
    auto vel =
        getGeomFromSA(Velocity<Fields<SpatialDim>>(), state);
    copyGeomFromMesh("velocity", vel);
  }
  if (tags[vert_dim].count("force")) {
    copyGeomFromMesh("force", InternalForce<Fields>());
  }
  if (tags[vert_dim].count("nodal_mass")) {
    copyFromMesh("nodal_mass", NodalMass<Fields>());
  }
  if (tags[vert_dim].count("nodal_pressure")) {
    copyFromMesh(
        "nodal_pressure", FieldDB<array_type>::Self()["nodal pressure"]);
  }
  if (tags[vert_dim].count("mass")) {
    copyFromMesh("mass", NodalMass<Fields>());
  }
  if (tags[vert_dim].count("potential")) {
    copyFromMesh("potential", ElectricPotential<Fields>());
  }

  // Element data
  if (tags[elem_dim].count("mass_density")) {
    auto dens = getFromSA(MassDensity<Fields>(), state);
    copyElemScalarFromMesh("mass_density", dens);
  }
  if (tags[elem_dim].count("spatialDensity")) {
    auto dens = getFromSA(MassDensity<Fields>(), state);
    copyElemScalarFromMesh("spatialDensity", dens);
  }
  if (tags[elem_dim].count("userMatID")) {
    copyElemScalarFromMesh("userMatID", UserMatID<Fields>());
  }
  if (tags[elem_dim].count("material")) {
    copyElemScalarFromMesh("material", UserMatID<Fields>());
  }
  if (tags[elem_dim].count("time_step")) {
    copyElemScalarFromMesh("time_step", ElementTimeStep<Fields>());
  }
  if (tags[elem_dim].count("internal_energy_density")) {
    copyElemScalarFromMesh(
        "internal_energy_density", InternalEnergyDensity<Fields>());
  }
  if (tags[elem_dim].count("internal_energy_per_mass")) {
    auto e_over_m = getFromSA(InternalEnergyPerUnitMass<Fields>(), state);
    copyElemScalarFromMesh("internal_energy_per_mass", e_over_m);
  }
  if (tags[elem_dim].count("internal_energy")) {
    copyElemScalarFromMesh("internal_energy", ElementInternalEnergy<Fields>());
  }
  if (tags[elem_dim].count("joule_energy")) {
    copyElemScalarFromMesh("joule_energy", ElementJouleEnergy<Fields>());
  }
  if (tags[elem_dim].count("mass")) {
    copyElemScalarFromMesh("mass", ElementMass<Fields>());
  }
  if (tags[elem_dim].count("conductivity")) {
    copyElemScalarFromMesh("conductivity", Conductivity<Fields>());
  }
  if (tags[elem_dim].count("deformation_gradient")) {
    copyElemTensorFromMesh(
        "deformation_gradient", DeformationGradient<Fields>());
  }
  if (tags[elem_dim].count("fine_scale_displacement")) {
    copyGeomFromMesh(
        SpatialDim, "fine_scale_displacement",
        getGeomFromSA(FineScaleDisplacement<Fields>(), state));
  }
}

template <int SpatialDim>
void Fields<SpatialDim>::cleanTagsFromMesh(
    Omega_h::TagSet const& tags) const {
  auto mesh = femesh.omega_h_mesh;
  auto elem_dim = size_t(SpatialDim);
  auto vert_dim = size_t(0);

  // Node data
  if (tags[vert_dim].count("vel")) mesh->remove_tag(vert_dim, "vel");
  if (tags[vert_dim].count("velocity")) mesh->remove_tag(vert_dim, "velocity");
  if (tags[vert_dim].count("force")) mesh->remove_tag(vert_dim, "force");
  if (tags[vert_dim].count("nodal_mass"))
    mesh->remove_tag(vert_dim, "nodal_mass");
  if (tags[vert_dim].count("nodal_pressure"))
    mesh->remove_tag(vert_dim, "nodal_pressure");
  if (tags[vert_dim].count("mass")) mesh->remove_tag(vert_dim, "mass");
  if (tags[vert_dim].count("potential"))
    mesh->remove_tag(vert_dim, "potential");

  // Element data
  if (tags[elem_dim].count("mass_density")) {
    mesh->remove_tag(mesh->dim(), "mass_density");
  }
  if (tags[elem_dim].count("spatialDensity")) {
    mesh->remove_tag(mesh->dim(), "spatialDensity");
  }
  if (tags[elem_dim].count("userMatID"))
    mesh->remove_tag(mesh->dim(), "userMatID");
  if (tags[elem_dim].count("material"))
    mesh->remove_tag(mesh->dim(), "material");
  if (tags[elem_dim].count("time_step"))
    mesh->remove_tag(mesh->dim(), "time_step");
  if (tags[elem_dim].count("internal_energy_density")) {
    mesh->remove_tag(mesh->dim(), "internal_energy_density");
  }
  if (tags[elem_dim].count("internal_energy_per_mass")) {
    mesh->remove_tag(mesh->dim(), "internal_energy_per_mass");
  }
  if (tags[elem_dim].count("internal_energy")) {
    mesh->remove_tag(mesh->dim(), "internal_energy");
  }
  if (tags[elem_dim].count("joule_energy")) {
    mesh->remove_tag(mesh->dim(), "joule_energy");
  }
  if (tags[elem_dim].count("mass")) mesh->remove_tag(mesh->dim(), "mass");
  if (tags[elem_dim].count("conductivity"))
    mesh->remove_tag(mesh->dim(), "conductivity");
  if (tags[elem_dim].count("deformation_gradient")) {
    mesh->remove_tag(mesh->dim(), "deformation_gradient");
  }
//  if (tags[elem_dim].count("stress")) {
//    mesh->remove_tag(mesh->dim(), "stress");
//  }
  if (tags[elem_dim].count("fine_scale_displacement")) {
    mesh->remove_tag(mesh->dim(), "fine_scale_displacement");
  }
}

template struct Fields<1>;
template struct Fields<2>;
template struct Fields<3>;

} /* namespace lgr */
