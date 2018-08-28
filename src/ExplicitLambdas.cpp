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

#include "ExplicitLambdas.hpp"
#include "FieldDB.hpp"
#include "LGRLambda.hpp"

namespace lgr {

template <int SpatialDim>
void update_elements_after_remap(
    const lgr::Fields<SpatialDim> &mesh_fields,
    const int                                            remapState) {
  typedef lgr::Fields<SpatialDim> Fields;
  auto mass_density = MassDensity<Fields>();
  auto elem_mass = ElementMass<Fields>();
  auto elem_energy = ElementInternalEnergy<Fields>();
  auto elem_volume = ElementVolume<Fields>();
  auto internal_energy_per_unit_mass = InternalEnergyPerUnitMass<Fields>();
  auto internal_energy_density = InternalEnergyDensity<Fields>();
  auto F = DeformationGradient<Fields>();
  auto Fold = FieldDB<typename Fields::elem_tensor_type>::Self().at(
      "save the deformation gradient");
  auto state = remapState;
  auto f = LAMBDA_EXPRESSION(int ielem) {
    elem_mass(ielem) = mass_density(ielem, state) * elem_volume(ielem);
    elem_energy(ielem) = internal_energy_density(ielem) * elem_volume(ielem);
    internal_energy_per_unit_mass(ielem, state) =
        elem_energy(ielem) / elem_mass(ielem);
    for (int ii = 0; ii < 9; ++ii) Fold(ielem, ii) = F(ielem, ii);
  };
  Kokkos::parallel_for(mesh_fields.femesh.nelems, f);
};

#define LGR_EXPL_INST(SpatialDim) \
template \
void update_elements_after_remap( \
    const lgr::Fields<SpatialDim> &mesh_fields, \
    const int                                            remapState);
LGR_EXPL_INST(3)
LGR_EXPL_INST(2)
#undef LGR_EXPL_INST

}  // namespace lgr
