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

#include "FieldsEnum.hpp"

#include "ElementHelpers.hpp"

#include "FieldDB.hpp"
#include "LagrangianFineScale.hpp"

namespace lgr {

template <int SpatialDim>
LagrangianFineScale<SpatialDim>::LagrangianFineScale(
      Fields &mesh_fields, int state0_in, int state1_in, Scalar c_tau_in)
      : elem_node_connectivity(mesh_fields.femesh.elem_node_ids)
      , updatedCoordinates(Coordinates<Fields>())
      , nodal_volume(NodalVolume<Fields>())
      , nodal_mass(NodalMass<Fields>())
      , elem_mass(ElementMass<Fields>())
      , pprime(FineScalePressure<Fields>())
      , uprime(FineScaleDisplacement<Fields>())
      , vprime(FineScaleVelocity<Fields>())
      , bulkModulus(BulkModulus<Fields>())
      , planeWaveModulus(PlaneWaveModulus<Fields>())
      , nodal_pressure(NodalPressure<Fields>())
      , nodal_pressure_increment(NodalPressureIncrement<Fields>())
      , state0_(state0_in)
      , state1_(state1_in)
      , c_tau_(c_tau_in)
      , nelems_(mesh_fields.femesh.nelems) {
    velocity[0] = Fields::getGeomFromSA(Velocity<Fields>(), state0_in);
    velocity[1] = Fields::getGeomFromSA(Velocity<Fields>(), state1_in);
  }

template <int SpatialDim>
  KOKKOS_INLINE_FUNCTION
  void LagrangianFineScale<SpatialDim>::operator()(int ielem) const {
    Scalar x[ElemNodeCount], y[ElemNodeCount], z[ElemNodeCount];
    Scalar vx[ElemNodeCount], vy[ElemNodeCount], vz[ElemNodeCount];
    Scalar grad_x[ElemNodeCount], grad_y[ElemNodeCount], grad_z[ElemNodeCount];
    Scalar vdot[] = {0.0, 0.0, 0.0};
    Scalar pressure[ElemNodeCount];
    Scalar pdot = 0.0;
    Scalar pCell = 0.0;
    typename Fields::geom_array_type oc(
        Fields::getGeomFromSA(updatedCoordinates, state0_));
    typename Fields::geom_array_type nc(
        Fields::getGeomFromSA(updatedCoordinates, state1_));
    for (int i = 0; i < ElemNodeCount; ++i) {
      const int n = elem_node_connectivity(ielem, i);

      x[i] = 0.5 * (oc(n, 0) + nc(n, 0));
      y[i] = 0.5 * (oc(n, 1) + nc(n, 1));
      z[i] = 0.5 * (oc(n, 2) + nc(n, 2));

      vx[i] = 0.5 * velocity[state0_](n, 0) + 0.5 * velocity[state1_](n, 0);
      vy[i] = 0.5 * velocity[state0_](n, 1) + 0.5 * velocity[state1_](n, 1);
      vz[i] = 0.5 * velocity[state0_](n, 2) + 0.5 * velocity[state1_](n, 2);

      vdot[0] += velocity[state1_](n, 0) - velocity[state0_](n, 0);
      vdot[1] += velocity[state1_](n, 1) - velocity[state0_](n, 1);
      vdot[2] += velocity[state1_](n, 2) - velocity[state0_](n, 2);

      pCell += nodal_pressure(n);
      pressure[i] = nodal_pressure(n);
      pdot += nodal_pressure_increment(n);
    }
    for (int slot = 0; slot < Fields::SpaceDim; ++slot) {
      vdot[slot] /= ElemNodeCount;
      vdot[slot] /= dt_;
    }
    pCell /= ElemNodeCount;
    pdot /= ElemNodeCount;
    pdot /= dt_;

    comp_grad(x, y, z, grad_x, grad_y, grad_z);
    const Scalar elem_volume = dot4(x, grad_x);

    const Scalar dil = 0.5 * (planeWaveModulus(ielem, state0_) +
                              planeWaveModulus(ielem, state1_));
    const Scalar rho = elem_mass(ielem) / elem_volume;
    const Scalar c =
        (dil / rho > 0.0)
            ? sqrt(dil / rho)
            : 1e-16;  // clip to make sure we don't take sqrt of negative...

    //const Scalar hvol = pow(elem_volume,1./3.);
    //const Scalar hart = maxEdgeLength(x, y, z);
    const Scalar factor = 1.0; // = 2/sqrt(NumNodes) = 2/sqrt(4)
    const Scalar colon = ( dot<4>( grad_x , grad_x ) +
         dot<4>( grad_y , grad_y ) +
         dot<4>( grad_z , grad_z ) );
    const Scalar h = factor * elem_volume / sqrt(colon);
    const Scalar tau = (c_tau_* h) / ( 2.0*c );

    Scalar gradp[3];
    gradp[0] = dot<ElemNodeCount>(grad_x, pressure) / elem_volume;
    gradp[1] = dot<ElemNodeCount>(grad_y, pressure) / elem_volume;
    if(SpatialDim == 3) gradp[2] = dot<ElemNodeCount>(grad_z, pressure) / elem_volume;

    const Scalar K =
        0.5 * (bulkModulus(ielem, state0_) + bulkModulus(ielem, state1_));

    for (int slot = 0; slot < Fields::SpaceDim; ++slot) {
      vprime(ielem, slot) = -(tau / rho) * (rho * vdot[slot] + gradp[slot]);
      uprime(ielem, slot, state1_) =
          uprime(ielem, slot, state0_) + dt_ * K * vprime(ielem, slot);
    }

    const Scalar divv = (dot4(grad_x, vx) +
                         dot4(grad_y, vy) +
                         dot4(grad_z, vz)) /
                        elem_volume;

    pprime(ielem) = -tau * (pdot + K * divv);

    return;
  }

template <int SpatialDim>
  void LagrangianFineScale<SpatialDim>::apply(Scalar timeStep) {
    this->dt_ = timeStep;
    Kokkos::parallel_for(nelems_, *this);
  }

template class LagrangianFineScale<3>;
template class LagrangianFineScale<2>;

}  //end namespace lgr
