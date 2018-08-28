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

#include "LagrangianTimeIntegration.hpp"
#include "LagrangianNodalPressure.hpp"
#include "ExplicitFunctors.hpp"
#include "LagrangianFineScale.hpp"
#include "FieldDB.hpp"
#include "LGRLambda.hpp"
#include <Kokkos_Timer.hpp>

namespace lgr {

PerformanceData::PerformanceData()
    : mesh_time(0)
    , init_time(0)
    , internal_force_time(0)
    , midpoint(0)
    , comm_time(0)
    , number_of_steps(0) {}

void PerformanceData::best(const PerformanceData &rhs) {
  if (rhs.mesh_time < mesh_time) mesh_time = rhs.mesh_time;
  if (rhs.init_time < init_time) init_time = rhs.init_time;
  if (rhs.internal_force_time < internal_force_time)
    internal_force_time = rhs.internal_force_time;
  if (rhs.midpoint < midpoint) midpoint = rhs.midpoint;
  if (rhs.comm_time < comm_time) comm_time = rhs.comm_time;
}

template <int SpatialDim>
LagrangianStep<SpatialDim>::LagrangianStep(
      std::list<std::shared_ptr<
          MaterialModelBase<SpatialDim>>>
          &          material_models,
      Fields &       mesh_fields,
      comm::Machine  machine,
      Omega_h::Mesh *mesh)
      : theMaterialModels_(material_models)
      , meshFields_(mesh_fields)
      , machine_(machine)
      , mesh_(mesh) {}

template <int SpatialDim>
PerformanceData LagrangianStep<SpatialDim>::advanceTime(
    const VectorContributions<SpatialDim>& accel_contribs,
    const VectorContributions<SpatialDim>& internal_force_contribs,
    const Scalar                      simtime,
    const Scalar                      dt,
    const int                         current_state,
    const int                         next_state) const {
  PerformanceData     perfData;
  Kokkos::Timer wall_clock;
  wall_clock.reset();

  //initialize time step nodes
  {
    const typename Fields::geom_array_type cur_vel(
        Fields::getGeomFromSA(Velocity<Fields>(), current_state));
    const typename Fields::geom_array_type next_vel(
        Fields::getGeomFromSA(Velocity<Fields>(), next_state));
    const typename Fields::geom_array_type disp(Displacement<Fields>());
    const typename Fields::geom_array_type xn(
        Fields::getGeomFromSA(Coordinates<Fields>(), current_state));
    const typename Fields::geom_array_type xnp1(
        Fields::getGeomFromSA(Coordinates<Fields>(), next_state));
    Kokkos::deep_copy(next_vel, cur_vel);
    Kokkos::deep_copy(xnp1, xn);
    Kokkos::deep_copy(disp, 0.0);
  }
  //zero nodal pressure data
  {
    LagrangianNodalPressure<SpatialDim> lnp(
        meshFields_, current_state, next_state);
    lnp.zeroData();
  }

  initialize_time_step_elements<SpatialDim>::apply(
      meshFields_, current_state, next_state);

  // get VMS stabilization parameter
  Teuchos::ParameterList &fieldData = meshFields_.fieldData;
  Scalar c_tau = fieldData.get<double>("vms stabilization parameter", 1.0);
  //compute fine scale fields
  {
    LagrangianFineScale<SpatialDim> lfs(
        meshFields_, current_state, next_state, c_tau);
    lfs.apply(dt);
  }

  //compute/initialize nodal pressure before beginning fixed point iteration
  {
    LagrangianNodalPressure<SpatialDim> lnp(
        meshFields_, current_state, next_state);
    lnp.computeNodalPressure();
  }

  perfData.internal_force_time = 0.0;
  perfData.comm_time = 0.0;
  for (int iterationCount = 0; iterationCount < 2; ++iterationCount) {
    //volume, gradient, velocity gradient, mid-configuration x_{n+1/2}.
    //the artificial viscosity uses the velocity gradient.
    {
      const Scalar alpha(0.5);
      grad<SpatialDim>::apply(
          meshFields_, current_state, next_state, alpha);
    }

    //calculate and store internal forces for each element.
    {
      const double t0 = wall_clock.seconds();
      internal_force<SpatialDim>::apply(
          meshFields_, current_state, next_state);
      const double t1 = wall_clock.seconds();
      perfData.internal_force_time += comm::max(machine_, t1 - t0);
    }
    execution_space::fence();

    //Assemble element contributions to nodal force into a nodal force vector.
    assemble_forces<SpatialDim>::apply(meshFields_);

    // Apply force-based boundary conditions
    internal_force_contribs.add_to(InternalForce<Fields>());

    //mpi swap and add nodal forces
    {
      const double t0 = wall_clock.seconds();
      meshFields_.conformGeom("force", InternalForce<Fields>());
      const double t1 = wall_clock.seconds();
      perfData.comm_time += comm::max(machine_, t1 - t0);
    }

    //compute acceleration
    {
      const typename Fields::array_type &nodal_mass = NodalMass<Fields>();
      const typename Fields::geom_array_type &acceleration =
          Acceleration<Fields>();
      const typename Fields::geom_array_type &internal_force =
          InternalForce<Fields>();
      auto updateAcceleration =
          LAMBDA_EXPRESSION(int inode) {
        const Scalar m = nodal_mass(inode);
        for (int slot = 0; slot < 3; ++slot)
          acceleration(inode, slot) = -(internal_force(inode, slot) / m);
      };  //end lambda updateAcceleration
      Kokkos::parallel_for(meshFields_.femesh.nnodes, updateAcceleration);
    }

    //Apply zero acceleration boundary conditions
    accel_contribs.add_to(Acceleration<Fields>());

    //update velocity
    {
      const typename Fields::geom_array_type cur_vel(
          Fields::getGeomFromSA(Velocity<Fields>(), current_state));
      const typename Fields::geom_array_type next_vel(
          Fields::getGeomFromSA(Velocity<Fields>(), next_state));
      const typename Fields::geom_array_type acceleration(
          Acceleration<Fields>());
      auto updateVelocity = LAMBDA_EXPRESSION(int inode) {
        const Scalar dt_vel = dt;
        for (int slot = 0; slot < 3; ++slot) {
          next_vel(inode, slot) =
              cur_vel(inode, slot) + dt_vel * acceleration(inode, slot);
        }
      };  //end lambda updateVelocity
      Kokkos::parallel_for(meshFields_.femesh.nnodes, updateVelocity);
    }

    //mpi conform nodal velocity
    {
      const double t0 = wall_clock.seconds();
      meshFields_.conformGeom(
          "vel", Fields::getGeomFromSA(Velocity<Fields>(), next_state));
      const double t1 = wall_clock.seconds();
      perfData.comm_time += comm::max(machine_, t1 - t0);
    }

    //update element internal energy
    energy_step<SpatialDim>::apply(
        meshFields_, dt, current_state, next_state);

    //update coordinates
    {
      const typename Fields::geom_array_type xn(
          Fields::getGeomFromSA(Coordinates<Fields>(), current_state));
      const typename Fields::geom_array_type xnp1(
          Fields::getGeomFromSA(Coordinates<Fields>(), next_state));
      const typename Fields::geom_array_type cur_vel(
          Fields::getGeomFromSA(Velocity<Fields>(), current_state));
      const typename Fields::geom_array_type next_vel(
          Fields::getGeomFromSA(Velocity<Fields>(), next_state));
      const typename Fields::geom_array_type cur_disp(Displacement<Fields>());
      auto updateCoordinates = LAMBDA_EXPRESSION(int inode) {
        const Scalar dt_disp = dt;
        for (int slot = 0; slot < 3; ++slot) {
          const Scalar vel =
              0.5 * (cur_vel(inode, slot) + next_vel(inode, slot));
          cur_disp(inode, slot) = dt_disp * vel;
          xnp1(inode, slot) = xn(inode, slot) + cur_disp(inode, slot);
        }
      };  //end lambda updateCoordinates
      Kokkos::parallel_for(meshFields_.femesh.nnodes, updateCoordinates);
    }

    /*
      update gradients (including element volume, velocity gradient, and deformation gradient F)
      this call puts all gradients at time and configuration (n+1)
      i.e.  this computes \frac{ d (field)_{n+1} }{d x_{n+1} }
      so the F and the velocity gradient are at (n+1).
      this is FINE for HYPER-elastic materials, since the stress at (n+1) depends only on F at (n+1).
      for HYPO-elastic materials, we want the velocity gradient at (n+1/2), so this is wrong.
      for HYPO models, the call arguments should be (meshFields_,current_state,next_state,0.5), which
      would be wrong for HYPER materials.
      perhaps the velocity gradient and deformation gradient calcuations should be separated;
      currently they are together.
      since we are not presently using HYPO models, let's leave this as is for now.
      the velocity gradient is not used to update HYPER material models, so at this stage
      in the algorithm it does not really matter what it is.

      volume, gradient, velocity gradient, deformation gradient F at end-configuration x_{n+1}.
    */
    {
      const Scalar alpha(1.0);
      grad<SpatialDim>::apply(
          meshFields_, current_state, next_state, alpha);
      //deformation gradient
      GRAD<SpatialDim>::apply(
          meshFields_, current_state, next_state, alpha);
    }

    /*
      update element data(volume,density,stress,...).
      this uses the pre-computed deformation gradient F, but not the velocity gradient,
      since all materials are currently HYPER-elastic.
    */
    element_step<SpatialDim> theElementStep(meshFields_);
    theElementStep.apply(meshFields_, next_state);

    for (auto matPtr : theMaterialModels_) {
      matPtr->updateElements(meshFields_, next_state, simtime, dt);
    }

    //compute fine scale fields
    {
      LagrangianFineScale<SpatialDim> lfs(
          meshFields_, current_state, next_state, c_tau);
      lfs.apply(dt);
    }

    //compute nodal pressure
    {
      LagrangianNodalPressure<SpatialDim> lnp(
          meshFields_, current_state, next_state);
      lnp.computeNodalPressure();
    }

    execution_space::fence();
  }  //end for (int iterationCount=0; iterationCount<2; ++iterationCount)

  perfData.midpoint = comm::max(machine_, wall_clock.seconds());

  perfData.number_of_steps = 1;
  return perfData;
}  //end function advanceTime

template class LagrangianStep<3>;
template class LagrangianStep<2>;

}  //end namespace lgr
