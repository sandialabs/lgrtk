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

#include "PhysicalConstants.hpp"

#include "ExplicitFunctors.hpp"
#include "ElementHelpers.hpp"
#include "FieldDB.hpp"
#include "LGRLambda.hpp"

#include <Omega_h_adj.hpp>
#include <Omega_h_mesh.hpp>
#include <Omega_h_align.hpp>

#include "TensorOperations_inline.hpp"


namespace lgr {

template <int SpatialDim>
ArtificialViscosity<SpatialDim>::ArtificialViscosity(const Fields &meshFields)
: elem_volume(ElementVolume<Fields>())
  , elem_mass(ElementMass<Fields>())
  , vel_grad(VelocityGradient<Fields>())
  , planeWaveModulus(PlaneWaveModulus<Fields>()) 
  , mhd(meshFields)  
{
    const Teuchos::ParameterList &fieldData = meshFields.fieldData;
    this->lin_bulk_visc = fieldData.get<double>("Linear Bulk Viscosity");
    this->quad_bulk_visc = fieldData.get<double>("Quadratic Bulk Viscosity");
}


template <int SpatialDim>
KOKKOS_INLINE_FUNCTION
Omega_h::Few<Scalar, lgr::Fields<SpatialDim>::SymTensorLength>
ArtificialViscosity<SpatialDim>::kineticViscosity(
        int           ielem,
        int           state0,
        int           state1,
        const Scalar *x,
        const Scalar *y,
        const Scalar *z ) const {

    //use average of pre-computed and stored plane wave modulus; do not re-compute.
    const Scalar dil = 
      0.5 * planeWaveModulus(ielem, state0) +
      0.5 * planeWaveModulus(ielem, state1) +
      mhd.elementAlfvenWaveModulus(ielem, x,y,z );

    const Scalar rho = elem_mass(ielem) / elem_volume(ielem);
    const Scalar c =  (dil / rho > 0.0) ? sqrt(dil / rho) : 1e-16;  // clip to make sure we don't take sqrt of negative...
    const Scalar h = maxEdgeLength(x, y, z);

    const Scalar traced = tensorOps::trace(ielem, vel_grad);

    const int    signTraced = (traced < 0.0) ? -1 : +1;
    const int    compression = (1 - signTraced) / 2;
    const Scalar nu = compression * h *
            (lin_bulk_visc * c + h * quad_bulk_visc * (signTraced * traced));

    const Scalar mu = rho * nu;

    Omega_h::Few<Scalar, Fields::SymTensorLength> q;

    static_assert(SpatialDim == 2 || SpatialDim == 3, "SpatialDim template parameter must be 2 or 3.");
    if(SpatialDim == 2) {
        q[FieldsEnum<2>::K_S_XX] =      mu*vel_grad(ielem, FieldsEnum<2>::K_F_XX);
        q[FieldsEnum<2>::K_S_YY] =      mu*vel_grad(ielem, FieldsEnum<2>::K_F_YY);
        q[FieldsEnum<2>::K_S_XY] = mu*0.5*(vel_grad(ielem, FieldsEnum<2>::K_F_XY) + vel_grad(ielem, FieldsEnum<2>::K_F_YX));
    } else if (SpatialDim == 3) {
        q[FieldsEnum<3>::K_S_XX] =      mu*vel_grad(ielem, FieldsEnum<3>::K_F_XX);
        q[FieldsEnum<3>::K_S_YY] =      mu*vel_grad(ielem, FieldsEnum<3>::K_F_YY);
        q[FieldsEnum<3>::K_S_ZZ] =      mu*vel_grad(ielem, FieldsEnum<3>::K_F_ZZ);
        q[FieldsEnum<3>::K_S_XY] = mu*0.5*(vel_grad(ielem, FieldsEnum<3>::K_F_XY) + vel_grad(ielem, FieldsEnum<3>::K_F_YX));
        q[FieldsEnum<3>::K_S_YZ] = mu*0.5*(vel_grad(ielem, FieldsEnum<3>::K_F_YZ) + vel_grad(ielem, FieldsEnum<3>::K_F_ZY));
        q[FieldsEnum<3>::K_S_ZX] = mu*0.5*(vel_grad(ielem, FieldsEnum<3>::K_F_ZX) + vel_grad(ielem, FieldsEnum<3>::K_F_XZ));
    }

    return  q;
}
template <int SpatialDim>
KOKKOS_INLINE_FUNCTION
Omega_h::Few<Scalar, SpatialDim> ArtificialViscosity<SpatialDim>::heatFlux(
        int           ielem,
        int           state0,
        int           state1,
        const Scalar *x,
        const Scalar *y,
        const Scalar *z,
        const Scalar *grad_x,
        const Scalar *grad_y,
        const Scalar *grad_z,
        const Scalar *nodalPressure) const {

    //use average of pre-computed and stored plane wave modulus; do not re-compute.
    const Scalar dil = 0.5 * planeWaveModulus(ielem, state0) +
            0.5 * planeWaveModulus(ielem, state1);
    const Scalar volume = elem_volume(ielem);
    const Scalar rho = elem_mass(ielem) / volume;
    const Scalar c = (dil / rho > 0.0) ? sqrt(dil / rho): 1e-16;  // clip to make sure we don't take sqrt of negative...
    const Scalar h = maxEdgeLength(x, y, z);

    const Scalar traced = tensorOps::trace(ielem, vel_grad);

    const int    signTraced = (traced < 0.0) ? -1 : +1;
    const int    compression = (1 - signTraced) / 2;
    const Scalar nu =
            compression * h *
            (lin_bulk_visc * c + h * quad_bulk_visc * (signTraced * traced));

    //the grad_x,grad_y,grad_z are already integrated over the element volume.
    //they are really the integrals of shape function gradients.
    //so we need to divide by volume here to compute a poinwise pressure gradient.
    static_assert(SpatialDim == 3 || SpatialDim == 2, "Template parameter SpatialDim must be 3 or 2");
    Scalar gradp[3] = {0,0,0};
    if(SpatialDim == 3) {
        gradp[0] = dot4(grad_x, nodalPressure) / volume;
        gradp[1] = dot4(grad_y, nodalPressure) / volume;
        gradp[2] = dot4(grad_z, nodalPressure) / volume;
    } else {
        gradp[0] =  dot3(grad_x, nodalPressure) / volume;
        gradp[1] =  dot3(grad_y, nodalPressure) / volume;
    }

    /*
      HACK
      hard-wired for ideal gas with gamma = 1.4
      HACK
     */
    const Scalar            gamma = 1.4;
    const Scalar            chi = gamma - 1.0;
    Omega_h::Few<Scalar, SpatialDim> lambda;
    for (int i = 0; i < SpatialDim; ++i) lambda[i] = -(nu / chi) * gradp[i];

    return lambda;
}

//----------------------------------------------------------------------------

template <int SpatialDim>
explicit_time_step<SpatialDim>::explicit_time_step(const Fields &meshFields)
: elem_node_connectivity(meshFields.femesh.elem_node_ids)
  , updatedCoordinates(Coordinates<Fields>())
  , elem_volume(ElementVolume<Fields>())
  , elem_mass(ElementMass<Fields>())
  , vel_grad(VelocityGradient<Fields>())
  , planeWaveModulus(PlaneWaveModulus<Fields>())
  , elem_time_step(ElementTimeStep<Fields>())
  , numElements(meshFields.femesh.nelems)
  , state(0) 
  , mhd(meshFields)
 {
    Teuchos::ParameterList &fieldData = const_cast<Teuchos::ParameterList&>(meshFields.fieldData);
    this->c_tau = fieldData.get<double>("vms stabilization parameter",1.);
    this->lin_bulk_visc = fieldData.get<double>("Linear Bulk Viscosity");
    this->quad_bulk_visc = fieldData.get<double>("Quadratic Bulk Viscosity");
 }

template <int SpatialDim>
KOKKOS_INLINE_FUNCTION
void explicit_time_step<SpatialDim>::init(value_type &update) const {
    update = Omega_h::ArithTraits<Scalar>::max();
}

template <int SpatialDim>
KOKKOS_INLINE_FUNCTION
void explicit_time_step<SpatialDim>::join(
        volatile value_type &update, const volatile value_type &source) const {
    update = update < source ? update : source;
}

template <int SpatialDim>
KOKKOS_INLINE_FUNCTION
Scalar explicit_time_step<SpatialDim>::elementTimeStep(
        int           ielem,
        const Scalar *x,
        const Scalar *y,
        const Scalar *z,
        const Scalar *grad_x,
        const Scalar *grad_y,
        const Scalar *grad_z) const {

    //use pre-computed and stored plane wave modulus; do not re-compute.
    const Scalar dil = planeWaveModulus(ielem,state) + mhd.elementAlfvenWaveModulus(ielem, x,y,z );
    const Scalar rho = elem_mass(ielem) / elem_volume(ielem);
    const Scalar c = (dil / rho > 0.0) ? sqrt(dil / rho) : 1e-16;  // clip to make sure we don't take sqrt of negative...
    const Scalar factor = 1.0; // = 2./sqrt(NumNodes) = 2./sqrt(4);
    if(SpatialDim != 3) Kokkos::abort("colon is wrong for 2d");
    const Scalar colon =  (dot4(grad_x, grad_x) + dot4(grad_y, grad_y) +   dot4(grad_z, grad_z));
    const Scalar h = factor * elem_volume(ielem) / sqrt(colon);
    const Scalar hOverC = h / c;
    const Scalar hart = maxEdgeLength(x, y, z);
    const Scalar hartOverC = hart / c;

    const Scalar traced = tensorOps::trace(ielem, vel_grad);

    Scalar xi = lin_bulk_visc + quad_bulk_visc * hartOverC * std::abs(traced);
    xi *= (hart / h);
    const Scalar Omega = 2.0 / (sqrt(c_tau + xi * xi) + xi);

    const Scalar timeStepEstimate = (Omega / 2.0) * hOverC;

    return timeStepEstimate;
}

template <int SpatialDim>
KOKKOS_INLINE_FUNCTION
void explicit_time_step<SpatialDim>::operator()(int ielem, value_type &update) const {
    const int ElemNodeCount = Fields::ElemNodeCount;
    Scalar    x[ElemNodeCount], y[ElemNodeCount], z[ElemNodeCount];
    Scalar grad_x[ElemNodeCount], grad_y[ElemNodeCount], grad_z[ElemNodeCount];

    typename Fields::geom_array_type cgeom(
            Fields::getGeomFromSA(updatedCoordinates, state));

    // Position:
    for (int i = 0; i < ElemNodeCount; ++i) {
        const int n = elem_node_connectivity(ielem, i);
        //use coordinates at next state
        x[i] = cgeom(n, 0);
        y[i] = cgeom(n, 1);
        z[i] = cgeom(n, 2);
    }

    // Gradient:
    comp_grad(x, y, z, grad_x, grad_y, grad_z);

    const Scalar dt_elem =
            this->elementTimeStep(ielem, x, y, z, grad_x, grad_y, grad_z);

    elem_time_step(ielem) = dt_elem;

    update = update < dt_elem ? update : dt_elem;
}

template <int SpatialDim>
Scalar explicit_time_step<SpatialDim>::apply(const int arg_state) {
    this->state = arg_state;
    Scalar dtStable = Omega_h::ArithTraits<Scalar>::max();
    Kokkos::parallel_reduce(this->numElements, *this, dtStable);
    return dtStable;
}

template <int SpatialDim>
initialize_element<SpatialDim>::initialize_element(
        const Fields &mesh_fields, const InitialConditions<Fields> &)
        : elem_node_connectivity(mesh_fields.femesh.elem_node_ids)
          , F(DeformationGradient<Fields>())
          , Fold(FieldDB<typename Fields::elem_tensor_type>::Self().at(
                  "save the deformation gradient"))
                  , elem_mass(ElementMass<Fields>())
                  , elem_energy(ElementInternalEnergy<Fields>())
                  , elem_volume(ElementVolume<Fields>())
                  , internal_energy_per_unit_mass(InternalEnergyPerUnitMass<Fields>())
                  , internal_energy_density(InternalEnergyDensity<Fields>())
                  , planeWaveModulus(PlaneWaveModulus<Fields>())
                  , mass_density(MassDensity<Fields>())
                  , model_coords(mesh_fields.femesh.node_coords) {}

template <int SpatialDim>
KOKKOS_INLINE_FUNCTION
void initialize_element<SpatialDim>::operator()(int ielem) const {
    const int K_XX = 0;
    const int K_YY = 1;
    const int K_ZZ = 2;

    Scalar x[Fields::ElemNodeCount];
    Scalar y[Fields::ElemNodeCount];
    Scalar z[Fields::ElemNodeCount];
    Scalar grad_x[Fields::ElemNodeCount];
    Scalar grad_y[Fields::ElemNodeCount];
    Scalar grad_z[Fields::ElemNodeCount];

    for (int i = 0; i < Fields::ElemNodeCount; ++i) {
        const int n = elem_node_connectivity(ielem, i);

        x[i] = model_coords(n, 0);
        y[i] = model_coords(n, 1);
        z[i] = model_coords(n, 2);
    }

    comp_grad(x, y, z, grad_x, grad_y, grad_z);

    if(SpatialDim != 3) {
      Kokkos::abort("explicit_functors needs fixed for 2d");
    }
    Fold(ielem, K_XX) = 1.;
    Fold(ielem, K_YY) = 1.;
    Fold(ielem, K_ZZ) = 1.;
    F(ielem, K_XX) = 1.;
    F(ielem, K_YY) = 1.;
    F(ielem, K_ZZ) = 1.;

    elem_volume(ielem) = dot4(x, grad_x);
    elem_mass(ielem) = mass_density(ielem, 0) * elem_volume(ielem);
    elem_energy(ielem) =
            elem_mass(ielem) * internal_energy_per_unit_mass(ielem, 0);
    internal_energy_density(ielem) = elem_energy(ielem) / elem_volume(ielem);
}

template <int SpatialDim>
void initialize_element<SpatialDim>::apply(
        const Fields &mesh_fields, const InitialConditions<Fields> &ic) {
    initialize_element op(mesh_fields, ic);
    Kokkos::parallel_for(mesh_fields.femesh.nelems, op);
}

template <int SpatialDim>
initialize_node<SpatialDim>::initialize_node(const Fields &mesh_fields)
: node_elem_connectivity(mesh_fields.femesh.node_elem_ids)
  , nodal_mass(NodalMass<Fields>())
  , elem_mass(ElementMass<Fields>())
  , model_coords(mesh_fields.femesh.node_coords)
  , coord_subview_state_0(Fields::getGeomFromSA(Coordinates<Fields>(), 0))
  , coord_subview_state_1(Fields::getGeomFromSA(Coordinates<Fields>(), 1)) {
    Kokkos::deep_copy(coord_subview_state_0, mesh_fields.femesh.node_coords);
    Kokkos::deep_copy(coord_subview_state_1, mesh_fields.femesh.node_coords);
}

template <int SpatialDim>
KOKKOS_INLINE_FUNCTION
void initialize_node<SpatialDim>::operator()(int inode) const {
    const int begin = node_elem_connectivity.row_map[inode];
    const int end = node_elem_connectivity.row_map[inode + 1];

    const int NUM_DIMS = 3;

    Scalar node_mass = 0;

    for (int i = begin; i != end; ++i) {
        const int elem_id = node_elem_connectivity.entries(i, 0);
        node_mass += elem_mass(elem_id);
    }

    nodal_mass(inode) = node_mass / ElemNodeCount;

    for (int slot = 0; slot < NUM_DIMS; ++slot) {
        coord_subview_state_0(inode, slot) = model_coords(inode, slot);
        coord_subview_state_1(inode, slot) = model_coords(inode, slot);
    }
}

template <int SpatialDim>
void initialize_node<SpatialDim>::apply(const Fields &mesh_fields) {
    initialize_node op(mesh_fields);
    Kokkos::parallel_for(mesh_fields.femesh.nnodes, op);
}

template <int SpatialDim>
update_node_mass_after_remap<SpatialDim>::update_node_mass_after_remap(const Fields &mesh_fields, int arg_state)
: node_elem_connectivity(mesh_fields.femesh.node_elem_ids)
  , nodal_mass(NodalMass<Fields>())
  , elem_mass(ElementMass<Fields>())
  , velocity(Fields::getGeomFromSA(Velocity<Fields>(), arg_state))
  , state(arg_state) {}

template <int SpatialDim>
KOKKOS_INLINE_FUNCTION
void update_node_mass_after_remap<SpatialDim>::operator()(int inode) const {
    const int begin = node_elem_connectivity.row_map[inode];
    const int end = node_elem_connectivity.row_map[inode + 1];
    Scalar    node_mass = 0;
    for (int i = begin; i != end; ++i) {
        const int elem_id = node_elem_connectivity.entries(i, 0);
        node_mass += elem_mass(elem_id);
    }
    nodal_mass(inode) = node_mass / ElemNodeCount;
}

template <int SpatialDim>
void update_node_mass_after_remap<SpatialDim>::apply(const Fields &mesh_fields, int arg_state) {
    update_node_mass_after_remap op(mesh_fields, arg_state);
    Kokkos::parallel_for(mesh_fields.femesh.nnodes, op);
}

template <int SpatialDim>
initialize_time_step_elements<SpatialDim>::initialize_time_step_elements(
        const Fields &, const int arg_state0, const int arg_state1)
        : mass_density(MassDensity<Fields>())
          , internal_energy_per_unit_mass(InternalEnergyPerUnitMass<Fields>())
          , planeWaveModulus(PlaneWaveModulus<Fields>())
          , bulkModulus(BulkModulus<Fields>())
          , stress(Stress<Fields>())
          , F(DeformationGradient<Fields>())
          , Fold(FieldDB<typename Fields::elem_tensor_type>::Self().at(
                  "save the deformation gradient"))
                  , uprime(FineScaleDisplacement<Fields>())
                  , state0(arg_state0)
                  , state1(arg_state1) {}

template <int SpatialDim>
void initialize_time_step_elements<SpatialDim>::apply(
        const Fields &mesh_fields, const int arg_state0, const int arg_state1) {
    initialize_time_step_elements op(mesh_fields, arg_state0, arg_state1);
    Kokkos::parallel_for(mesh_fields.femesh.nelems, op);
}

template <int SpatialDim>
KOKKOS_INLINE_FUNCTION
void initialize_time_step_elements<SpatialDim>::operator()(int ielem) const {
    mass_density(ielem, state1) = mass_density(ielem, state0);
    internal_energy_per_unit_mass(ielem, state1) =
            internal_energy_per_unit_mass(ielem, state0);
    planeWaveModulus(ielem, state1) = planeWaveModulus(ielem, state0);
    bulkModulus(ielem, state1) = bulkModulus(ielem, state0);
    for (int ii = 0; ii < 6; ++ii)
        stress(ielem, ii, state1) = stress(ielem, ii, state0);
    for (int ii = 0; ii < 9; ++ii) Fold(ielem, ii) = F(ielem, ii);
    for (int ii = 0; ii < Fields::SpaceDim; ++ii)
        uprime(ielem, ii, state1) = uprime(ielem, ii, state0);
    return;
}

template <int SpatialDim>
grad<SpatialDim>::grad(
        const Fields &fields,
        const int     arg_state0,
        const int     arg_state1,
        const Scalar  arg_alpha)
        : elem_node_connectivity(fields.femesh.elem_node_ids)
          , vel_grad(VelocityGradient<Fields>())
          , elem_volume(ElementVolume<Fields>())
          , state0(arg_state0)
          , state1(arg_state1)
          , alpha(arg_alpha) {
    velocity[0] = Fields::getGeomFromSA(Velocity<Fields>(), state0);
    velocity[1] = Fields::getGeomFromSA(Velocity<Fields>(), state1);
    xn = Fields::getGeomFromSA(Coordinates<Fields>(), state0);
    xnp1 = Fields::getGeomFromSA(Coordinates<Fields>(), state1);
}

//   Calculate Velocity Gradients
template <>
KOKKOS_INLINE_FUNCTION
void grad<3>::v_grad(
        int     ielem,
        Scalar *vx,
        Scalar *vy,
        Scalar *vz,
        Scalar *grad_x,
        Scalar *grad_y,
        Scalar *grad_z,
        Scalar  inv_vol) const {


    vel_grad(ielem, FieldsEnum<3>::K_F_XX) =
            inv_vol * dot4(vx, grad_x);
    vel_grad(ielem,FieldsEnum<3>::K_F_YX) =
            inv_vol * dot4(vy, grad_x);
    vel_grad(ielem,FieldsEnum<3>::K_F_ZX) =
            inv_vol * dot4(vz, grad_x);

    vel_grad(ielem, FieldsEnum<3>::K_F_XY) =
            inv_vol * dot4(vx, grad_y);
    vel_grad(ielem,FieldsEnum<3>::K_F_YY) =
            inv_vol * dot4(vy, grad_y);
    vel_grad(ielem,FieldsEnum<3>::K_F_ZY) =
            inv_vol * dot4(vz, grad_y);

    vel_grad(ielem,FieldsEnum<3>::K_F_XZ) =
            inv_vol * dot4(vx, grad_z);
    vel_grad(ielem,FieldsEnum<3>::K_F_YZ) =
            inv_vol * dot4(vy, grad_z);
    vel_grad(ielem, FieldsEnum<3>::K_F_ZZ) =
            inv_vol * dot4(vz, grad_z);
}

template <>
KOKKOS_INLINE_FUNCTION
void grad<2>::v_grad(
        int     ielem,
        Scalar *vx,
        Scalar *vy,
        Scalar *,
        Scalar *grad_x,
        Scalar *grad_y,
        Scalar *,
        Scalar  inv_vol) const {


    vel_grad(ielem, FieldsEnum<2>::K_F_XX) =
            inv_vol * dot3(vx, grad_x);
    vel_grad(ielem,FieldsEnum<2>::K_F_YX) =
            inv_vol * dot3(vy, grad_x);

    vel_grad(ielem, FieldsEnum<2>::K_F_XY) =
            inv_vol * dot3(vx, grad_y);
    vel_grad(ielem,FieldsEnum<2>::K_F_YY) =
            inv_vol * dot3(vy, grad_y);

}

template <int SpatialDim>
KOKKOS_INLINE_FUNCTION
void grad<SpatialDim>::operator()(int ielem) const {
    const int Xslot = 0;
    const int Yslot = 1;
    const int Zslot = 2;

    //  declare and reuse local data for frequently accessed data to
    //  reduce global memory reads and writes.

    Scalar X[ElemNodeCount], Y[ElemNodeCount], Z[ElemNodeCount];
    Scalar x[ElemNodeCount], y[ElemNodeCount], z[ElemNodeCount];

    Scalar xmid[ElemNodeCount], ymid[ElemNodeCount], zmid[ElemNodeCount];
    Scalar vx[ElemNodeCount], vy[ElemNodeCount], vz[ElemNodeCount];

    Scalar grad_x[ElemNodeCount], grad_y[ElemNodeCount], grad_z[ElemNodeCount];

    // Read global velocity once and use many times
    // via local registers / L1 cache.
    //  store the velocity information in local memory before using,
    //  so it can be returned for other functions to use

    // Read global coordinates and velocity once and use many times
    // via local registers / L1 cache.
    // load X coordinate information and move by half time step

    for (int i = 0; i < ElemNodeCount; ++i) {
        const int n = elem_node_connectivity(ielem, i);

        //updated reference configuration at time n
        X[i] = xn(n, Xslot);
        Y[i] = xn(n, Yslot);
        Z[i] = xn(n, Zslot);

        //configuration at time n+1
        x[i] = xnp1(n, Xslot);
        y[i] = xnp1(n, Yslot);
        z[i] = xnp1(n, Zslot);

        //average configuration
        xmid[i] = (1.0 - alpha) * X[i] + alpha * x[i];
        ymid[i] = (1.0 - alpha) * Y[i] + alpha * y[i];
        zmid[i] = (1.0 - alpha) * Z[i] + alpha * z[i];

        //average velocity
        vx[i] =
                (1.0 - alpha) * velocity[0](n, Xslot) + alpha * velocity[1](n, Xslot);
        vy[i] =
                (1.0 - alpha) * velocity[0](n, Yslot) + alpha * velocity[1](n, Yslot);
        vz[i] =
                (1.0 - alpha) * velocity[0](n, Zslot) + alpha * velocity[1](n, Zslot);
    }

    //  Calculate volume from x updatedCoordinates and gradient information

    //volume and velocity gradient
    comp_grad(
            xmid, ymid, zmid, grad_x, grad_y, grad_z);
    const Scalar vol = dot4(xmid, grad_x);
    elem_volume(ielem) = vol;
    const Scalar inv_vol = 1.0 / vol;
    v_grad(ielem, vx, vy, vz, grad_x, grad_y, grad_z, inv_vol);
}

template <int SpatialDim>
void grad<SpatialDim>::apply(
        const Fields &fields,
        const int     arg_state0,
        const int     arg_state1,
        const Scalar  arg_alpha) {
    grad op(fields, arg_state0, arg_state1, arg_alpha);
    Kokkos::parallel_for(fields.femesh.nelems, op);
}

template <int SpatialDim>
GRAD<SpatialDim>::GRAD(
        const Fields &fields,
        const int     arg_state0,
        const int     arg_state1,
        const Scalar  arg_alpha)
        : elem_node_connectivity(fields.femesh.elem_node_ids)
          , F(DeformationGradient<Fields>())
          , Fold(FieldDB<typename Fields::elem_tensor_type>::Self().at(
                  "save the deformation gradient"))
                  , state0(arg_state0)
                  , state1(arg_state1)
                  , alpha(arg_alpha) {
    xn = Fields::getGeomFromSA(Coordinates<Fields>(), state0);
    xnp1 = Fields::getGeomFromSA(Coordinates<Fields>(), state1);
}

//   Calculate deformation gradient
template <>
KOKKOS_INLINE_FUNCTION
void GRAD<3>::deformationGradient(
        int     ielem,
        Scalar *x,
        Scalar *y,
        Scalar *z,
        Scalar *GRAD_X,
        Scalar *GRAD_Y,
        Scalar *GRAD_Z,
        Scalar  inv_vol) const {
    constexpr int K_F_XX = FieldsEnum<3>::K_F_XX;
    constexpr int K_F_YY = FieldsEnum<3>::K_F_YY;
    constexpr int K_F_ZZ = FieldsEnum<3>::K_F_ZZ;
    constexpr int K_F_XY = FieldsEnum<3>::K_F_XY;
    constexpr int K_F_YZ = FieldsEnum<3>::K_F_YZ;
    constexpr int K_F_ZX = FieldsEnum<3>::K_F_ZX;
    constexpr int K_F_YX = FieldsEnum<3>::K_F_YX;
    constexpr int K_F_ZY = FieldsEnum<3>::K_F_ZY;
    constexpr int K_F_XZ = FieldsEnum<3>::K_F_XZ;

    //incremental deformation gradient
    Scalar f[9];
    f[K_F_XX] = inv_vol * dot4(x, GRAD_X);
    f[K_F_YX] = inv_vol * dot4(y, GRAD_X);
    f[K_F_ZX] = inv_vol * dot4(z, GRAD_X);

    f[K_F_XY] = inv_vol * dot4(x, GRAD_Y);
    f[K_F_YY] = inv_vol * dot4(y, GRAD_Y);
    f[K_F_ZY] = inv_vol * dot4(z, GRAD_Y);

    f[K_F_XZ] = inv_vol * dot4(x, GRAD_Z);
    f[K_F_YZ] = inv_vol * dot4(y, GRAD_Z);
    f[K_F_ZZ] = inv_vol * dot4(z, GRAD_Z);

    /*
    inline Tensor operator*(const Tensor& ta, const Tensor& tb)
    {
    return Tensor(
    F.xx = ta.xx*tb.xx + ta.xy*tb.yx + ta.xz*tb.zx,
    F.xy = ta.xx*tb.xy + ta.xy*tb.yy + ta.xz*tb.zy,
    F.xz = ta.xx*tb.xz + ta.xy*tb.yz + ta.xz*tb.zz,
    F.yx = ta.yx*tb.xx + ta.yy*tb.yx + ta.yz*tb.zx,
    F.yy = ta.yx*tb.xy + ta.yy*tb.yy + ta.yz*tb.zy,
    F.yz = ta.yx*tb.xz + ta.yy*tb.yz + ta.yz*tb.zz,
    F.zx = ta.zx*tb.xx + ta.zy*tb.yx + ta.zz*tb.zx,
    F.zy = ta.zx*tb.xy + ta.zy*tb.yy + ta.zz*tb.zy,
    F.zz = ta.zx*tb.xz + ta.zy*tb.yz + ta.zz*tb.zz );
    }
     */
    //multiplicatively update F = f*Fold
    F(ielem, K_F_XX) = f[K_F_XX] * Fold(ielem, K_F_XX) +
            f[K_F_XY] * Fold(ielem, K_F_YX) +
            f[K_F_XZ] * Fold(ielem, K_F_ZX);
    F(ielem, K_F_XY) = f[K_F_XX] * Fold(ielem, K_F_XY) +
            f[K_F_XY] * Fold(ielem, K_F_YY) +
            f[K_F_XZ] * Fold(ielem, K_F_ZY);
    F(ielem, K_F_XZ) = f[K_F_XX] * Fold(ielem, K_F_XZ) +
            f[K_F_XY] * Fold(ielem, K_F_YZ) +
            f[K_F_XZ] * Fold(ielem, K_F_ZZ);
    F(ielem, K_F_YX) = f[K_F_YX] * Fold(ielem, K_F_XX) +
            f[K_F_YY] * Fold(ielem, K_F_YX) +
            f[K_F_YZ] * Fold(ielem, K_F_ZX);
    F(ielem, K_F_YY) = f[K_F_YX] * Fold(ielem, K_F_XY) +
            f[K_F_YY] * Fold(ielem, K_F_YY) +
            f[K_F_YZ] * Fold(ielem, K_F_ZY);
    F(ielem, K_F_YZ) = f[K_F_YX] * Fold(ielem, K_F_XZ) +
            f[K_F_YY] * Fold(ielem, K_F_YZ) +
            f[K_F_YZ] * Fold(ielem, K_F_ZZ);
    F(ielem, K_F_ZX) = f[K_F_ZX] * Fold(ielem, K_F_XX) +
            f[K_F_ZY] * Fold(ielem, K_F_YX) +
            f[K_F_ZZ] * Fold(ielem, K_F_ZX);
    F(ielem, K_F_ZY) = f[K_F_ZX] * Fold(ielem, K_F_XY) +
            f[K_F_ZY] * Fold(ielem, K_F_YY) +
            f[K_F_ZZ] * Fold(ielem, K_F_ZY);
    F(ielem, K_F_ZZ) = f[K_F_ZX] * Fold(ielem, K_F_XZ) +
            f[K_F_ZY] * Fold(ielem, K_F_YZ) +
            f[K_F_ZZ] * Fold(ielem, K_F_ZZ);
}
template <>
KOKKOS_INLINE_FUNCTION
void GRAD<2>::deformationGradient(
        int     ielem,
        Scalar *x,
        Scalar *y,
        Scalar *,
        Scalar *GRAD_X,
        Scalar *GRAD_Y,
        Scalar *,
        Scalar  inv_vol) const {
    constexpr int K_F_XX = FieldsEnum<2>::K_F_XX;
    constexpr int K_F_YY = FieldsEnum<2>::K_F_YY;
    constexpr int K_F_XY = FieldsEnum<2>::K_F_XY;
    constexpr int K_F_YX = FieldsEnum<2>::K_F_YX;

    //incremental deformation gradient
    Scalar f[9];
    f[K_F_XX] = inv_vol * dot3(x, GRAD_X);
    f[K_F_YX] = inv_vol * dot3(y, GRAD_X);

    f[K_F_XY] = inv_vol * dot3(x, GRAD_Y);
    f[K_F_YY] = inv_vol * dot3(y, GRAD_Y);

    //multiplicatively update F = f*Fold
    F(ielem, K_F_XX) = f[K_F_XX] * Fold(ielem, K_F_XX) +
            f[K_F_XY] * Fold(ielem, K_F_YX);

    F(ielem, K_F_XY) = f[K_F_XX] * Fold(ielem, K_F_XY) +
            f[K_F_XY] * Fold(ielem, K_F_YY);

    F(ielem, K_F_YX) = f[K_F_YX] * Fold(ielem, K_F_XX) +
            f[K_F_YY] * Fold(ielem, K_F_YX);

    F(ielem, K_F_YY) = f[K_F_YX] * Fold(ielem, K_F_XY) +
            f[K_F_YY] * Fold(ielem, K_F_YY);

}

//--------------------------------------------------------------------------
// Functor operator() which calls the three member functions.

template <int SpatialDim>
KOKKOS_INLINE_FUNCTION
void GRAD<SpatialDim>::operator()(int ielem) const {
    const int Xslot = 0;
    const int Yslot = 1;
    const int Zslot = 2;

    //  declare and reuse local data for frequently accessed data to
    //  reduce global memory reads and writes.

    Scalar X[ElemNodeCount], Y[ElemNodeCount], Z[ElemNodeCount];
    Scalar x[ElemNodeCount], y[ElemNodeCount], z[ElemNodeCount];

    Scalar xmid[ElemNodeCount], ymid[ElemNodeCount], zmid[ElemNodeCount];

    Scalar GRAD_X[ElemNodeCount], GRAD_Y[ElemNodeCount], GRAD_Z[ElemNodeCount];

    // Read global velocity once and use many times
    // via local registers / L1 cache.
    //  store the velocity information in local memory before using,
    //  so it can be returned for other functions to use

    // Read global coordinates and velocity once and use many times
    // via local registers / L1 cache.
    // load X coordinate information and move by half time step

    for (int i = 0; i < ElemNodeCount; ++i) {
        const int n = elem_node_connectivity(ielem, i);

        //updated reference configuration at time n
        X[i] = xn(n, Xslot);
        Y[i] = xn(n, Yslot);
        Z[i] = xn(n, Zslot);

        //configuration at time n+1
        x[i] = xnp1(n, Xslot);
        y[i] = xnp1(n, Yslot);
        z[i] = xnp1(n, Zslot);

        //average configuration
        xmid[i] = (1.0 - alpha) * X[i] + alpha * x[i];
        ymid[i] = (1.0 - alpha) * Y[i] + alpha * y[i];
        zmid[i] = (1.0 - alpha) * Z[i] + alpha * z[i];
    }

    //deformation gradient
    comp_grad(X, Y, Z, GRAD_X, GRAD_Y, GRAD_Z);
    const Scalar VOL = dot4(X, GRAD_X);
    const Scalar INV_VOL = 1.0 / VOL;
    deformationGradient(
            ielem, xmid, ymid, zmid, GRAD_X, GRAD_Y, GRAD_Z, INV_VOL);
}

template <int SpatialDim>
void GRAD<SpatialDim>::apply(
        const Fields &fields,
        const int     arg_state0,
        const int     arg_state1,
        const Scalar  arg_alpha) {
    GRAD op(fields, arg_state0, arg_state1, arg_alpha);
    Kokkos::parallel_for(fields.femesh.nelems, op);
}

template <int SpatialDim>
internal_force<SpatialDim>::internal_force(
        const Fields &mesh_fields, const int arg_state0, const int arg_state1)
        : elem_node_connectivity(mesh_fields.femesh.elem_node_ids)
	, updatedCoordinates(Coordinates<Fields>())
	, elem_mass(ElementMass<Fields>())
	, stress(Stress<Fields>())
	, element_force(ElementForce<Fields>())
	, vel_grad(VelocityGradient<Fields>())
	, pprime(FineScalePressure<Fields>())
	, nodal_pressure(NodalPressure<Fields>())
	, state0(arg_state0)
	, state1(arg_state1)
	, artificialViscosityModel(mesh_fields) 
	, mhd(mesh_fields)
{
    velocity[0] = Fields::getGeomFromSA(Velocity<Fields>(), arg_state0);
    velocity[1] = Fields::getGeomFromSA(Velocity<Fields>(), arg_state1);
}

template <int SpatialDim>
void internal_force<SpatialDim>::apply(
        const Fields &mesh_fields, const int arg_state0, const int arg_state1) {
    internal_force op_force(mesh_fields, arg_state0, arg_state1);

    Kokkos::parallel_for(mesh_fields.femesh.nelems, op_force);
}

template <int SpatialDim>
KOKKOS_INLINE_FUNCTION
void internal_force<SpatialDim>::comp_force(
        int                 ielem,
        const Scalar *const grad_x,
        const Scalar *const grad_y,
        const Scalar *const grad_z,
        Scalar *            algoStress) const {

    for (int inode = 0; inode < ElemNodeCount; ++inode) {

        Scalar force[3] = {0,0,0};
         tensorOps::symmTimesVector<SpatialDim>(algoStress,
                 grad_x[inode], grad_y[inode], grad_z[inode], force);

         for(int d = 0; d < SpatialDim; ++d)
             element_force(ielem, d, inode) = force[d];

    }
}
  
template <int SpatialDim>
KOKKOS_INLINE_FUNCTION
void internal_force<SpatialDim>::operator()(int ielem) const {
    Scalar x[ElemNodeCount], y[ElemNodeCount], z[ElemNodeCount];
    Scalar grad_x[ElemNodeCount], grad_y[ElemNodeCount], grad_z[ElemNodeCount];

    typename Fields::geom_array_type oc(Fields::getGeomFromSA(updatedCoordinates, 0));
    typename Fields::geom_array_type nc(Fields::getGeomFromSA(updatedCoordinates, 1));

    Scalar ph(0.0);
    for (int i = 0; i < ElemNodeCount; ++i) {
        const int n = elem_node_connectivity(ielem, i);
        x[i] = 0.5 * (oc(n, 0) + nc(n, 0));
        y[i] = 0.5 * (oc(n, 1) + nc(n, 1));
        z[i] = 0.5 * (oc(n, 2) + nc(n, 2));
        ph += nodal_pressure(n);
    }
    ph /= ElemNodeCount;

    //Gradient:
    comp_grad(x, y, z, grad_x, grad_y, grad_z);

    //use stored vel_grad object to compute bulk viscosity
    const Omega_h::Few<Scalar, Fields::SymTensorLength> q = 
      artificialViscosityModel.kineticViscosity( ielem, 
						 state0, 
						 state1, 
						 x, y, z ); 
    
    //use stored stress to compute internal force
    Scalar total_stress[6];
    tensorOps::avgTensorState(ielem, state0, state1, stress, total_stress);

    deviatoricProjection<SpatialDim>(total_stress);
    Scalar newPressure = ph + pprime(ielem);
    tensorOps::symm_axpypc<SpatialDim>(1, total_stress, q.data(), -newPressure);

    typedef Omega_h::Matrix<3,3> Tensor;
    const Tensor magneticSigma = mhd.elementMagneticStressTensor(ielem, x,y,z );
    total_stress[FieldsEnum<3>::K_S_XX] += magneticSigma(0,0);
    total_stress[FieldsEnum<3>::K_S_YY] += magneticSigma(1,1);
    total_stress[FieldsEnum<3>::K_S_ZZ] += magneticSigma(2,2);
    total_stress[FieldsEnum<3>::K_S_XY] += 0.5*(magneticSigma(0,1) + magneticSigma(1,0));
    total_stress[FieldsEnum<3>::K_S_YZ] += 0.5*(magneticSigma(1,2) + magneticSigma(2,1));
    total_stress[FieldsEnum<3>::K_S_ZX] += 0.5*(magneticSigma(2,0) + magneticSigma(0,2));

    comp_force(ielem, grad_x, grad_y, grad_z, total_stress);
}

template <int SpatialDim>
energy_step<SpatialDim>::energy_step(
        const Fields &mesh_fields,
        const Scalar  arg_dt,
        const int     arg_state0,
        const int     arg_state1)
        : elem_node_connectivity(mesh_fields.femesh.elem_node_ids)
          , updatedCoordinates(Coordinates<Fields>())
          , elem_mass(ElementMass<Fields>())
          , elem_energy(ElementInternalEnergy<Fields>())
          , elem_volume(ElementVolume<Fields>())
          , internal_energy_per_unit_mass(InternalEnergyPerUnitMass<Fields>())
          , internal_energy_density(InternalEnergyDensity<Fields>())
          , stress(Stress<Fields>())
          , element_force(ElementForce<Fields>())
          , vel_grad(VelocityGradient<Fields>())
          , pprime(FineScalePressure<Fields>())
          , nodal_pressure(NodalPressure<Fields>())
          , uprime(FineScaleDisplacement<Fields>())
          , shockHeatFlux(ElementShockHeatFlux<Fields>())
          , dt_vel(arg_dt)
          , state0(arg_state0)
          , state1(arg_state1)
          , artificialViscosityModel(mesh_fields) {
    velocity[0] = Fields::getGeomFromSA(Velocity<Fields>(), arg_state0);
    velocity[1] = Fields::getGeomFromSA(Velocity<Fields>(), arg_state1);
}

template <int SpatialDim>
void energy_step<SpatialDim>::apply(
        const Fields &mesh_fields,
        const Scalar  arg_dt,
        const int     arg_state0,
        const int     arg_state1) {
    energy_step op_work(mesh_fields, arg_dt, arg_state0, arg_state1);

    Kokkos::parallel_for(mesh_fields.femesh.nelems, op_work);
}

template <int SpatialDim>
KOKKOS_INLINE_FUNCTION
    Scalar energy_step<SpatialDim>::comp_work(
            int,
            const Scalar *      vx,
            const Scalar *      vy,
            const Scalar *      vz,
            const Scalar *const grad_x,
            const Scalar *const grad_y,
            const Scalar *const grad_z,
            const Scalar *      algoStress) const {

        Scalar element_power = 0.0;
        for (int inode = 0; inode < ElemNodeCount; ++inode) {

            Scalar force[3];
            tensorOps::symmTimesVector<SpatialDim>(algoStress,
                    grad_x[inode], grad_y[inode], grad_z[inode], force);

            element_power += tensorOps::dot<SpatialDim>(force, vx[inode], vy[inode], vz[inode]);//swb this order is different than votd
        }

        const Scalar element_work = dt_vel * element_power;
        return element_work;
    }

template <int SpatialDim>
KOKKOS_INLINE_FUNCTION
void energy_step<SpatialDim>::operator()(int ielem) const {
    Scalar x[ElemNodeCount], y[ElemNodeCount], z[ElemNodeCount];
    Scalar grad_x[ElemNodeCount], grad_y[ElemNodeCount], grad_z[ElemNodeCount];
    Scalar pnode[ElemNodeCount];

    typename Fields::geom_array_type oc(
            Fields::getGeomFromSA(updatedCoordinates, 0));
    typename Fields::geom_array_type nc(
            Fields::getGeomFromSA(updatedCoordinates, 1));

    Scalar ph(0.0);
    for (int i = 0; i < ElemNodeCount; ++i) {
        const int n = elem_node_connectivity(ielem, i);
        x[i] = 0.5 * (oc(n, 0) + nc(n, 0));
        y[i] = 0.5 * (oc(n, 1) + nc(n, 1));
        z[i] = 0.5 * (oc(n, 2) + nc(n, 2));
        pnode[i] = nodal_pressure(n);
        ph += pnode[i];
    }
    ph /= ElemNodeCount;

    // Gradient:
    comp_grad(x, y, z, grad_x, grad_y, grad_z);

    //use stored vel_grad object to compute bulk viscosity
    const Omega_h::Few<Scalar, Fields::SymTensorLength> q = artificialViscosityModel.kineticViscosity(
            ielem, state0, state1, x, y, z);

    //compute artificial shock-based heat flux
    Omega_h::Few<Scalar, SpatialDim> lambda = artificialViscosityModel.heatFlux(
            ielem, state0, state1, x, y, z, grad_x, grad_y, grad_z, pnode);

    //add (ph*vprime) term to heat flux and store for future use
    for (int ii = 0; ii < SpatialDim; ++ii)
        shockHeatFlux(ielem, ii) =
                lambda[ii] +
                ph * (uprime(ielem, ii, state1) - uprime(ielem, ii, state0)) / dt_vel;

    Scalar total_stress[Fields::SymTensorLength ];

    tensorOps::avgTensorState(ielem, state0, state1, stress, total_stress);

    deviatoricProjection<SpatialDim>(total_stress);
    Scalar newPressure = ph + pprime(ielem);

    tensorOps::symm_axpypc<SpatialDim>(1, total_stress, q.data(), -newPressure);

    // velocity:
    Scalar vx[ElemNodeCount], vy[ElemNodeCount], vz[ElemNodeCount];
    for (int i = 0; i < ElemNodeCount; ++i) {
        const int n = elem_node_connectivity(ielem, i);
        //use velocity average of the two states

        tensorOps::avgVectors(n, i, velocity[0], velocity[1], vx, vy, vz);

    }

    const Scalar element_work =
            comp_work(ielem, vx, vy, vz, grad_x, grad_y, grad_z, total_stress);
    const Scalar deltae = element_work / elem_mass(ielem);
    internal_energy_per_unit_mass(ielem, state1) =
            internal_energy_per_unit_mass(ielem, state0) + deltae;
    elem_energy(ielem) =
            elem_mass(ielem) * internal_energy_per_unit_mass(ielem, state1);
    internal_energy_density(ielem) = elem_energy(ielem) / elem_volume(ielem);
}

template <int SpatialDim>
void shock_heat_flux_step<SpatialDim>::apply(
        Omega_h::Mesh *mesh,
        Fields &,
        const int      state,
        const Scalar   dt) {
    Omega_h::Adj elems2faces = mesh->ask_down(SpatialDim, SpatialDim - 1);
    Omega_h::Adj faces2elems = mesh->ask_up(SpatialDim - 1, SpatialDim);
    Omega_h::LOs faces2verts = mesh->ask_verts_of(SpatialDim - 1);
    Omega_h::LOs face_uses2faces = elems2faces.ab2b;
    Omega_h::Read<signed char> face_use_codes = elems2faces.codes;
    Omega_h::LOs               faces2elem_uses = faces2elems.a2ab;
    Omega_h::LOs               elem_uses2elems = faces2elems.ab2b;
    Omega_h::Read<signed char> elem_use_codes = faces2elems.codes;

    typename Fields::geom_state_array_type updatedCoordinates =
            Coordinates<Fields>();
    typename Fields::geom_array_type oc(
            Fields::getGeomFromSA(updatedCoordinates, 0));
    typename Fields::geom_array_type nc(
            Fields::getGeomFromSA(updatedCoordinates, 1));

    typename Fields::elem_vector_type shockHeatFlux =
            ElementShockHeatFlux<Fields>();

    typename Fields::array_type elem_mass = ElementMass<Fields>();
    typename Fields::array_type elem_energy = ElementInternalEnergy<Fields>();
    typename Fields::state_array_type internal_energy_per_unit_mass =
            InternalEnergyPerUnitMass<Fields>();

    auto diffuse = LAMBDA_EXPRESSION(int elem1) {

        const int nFacesPerElement(4);
        const int nNodesPerFace(3);

        //loop over faces attached to the element
        for (int elem1_face = 0; elem1_face < nFacesPerElement; ++elem1_face) {
            int face_use = (elem1 * nFacesPerElement) + elem1_face;
            int face = face_uses2faces[face_use];

            //compute area-weighted normal for a face
            Omega_h::Vector<3> face_pts[3];
            for (int face_vert = 0; face_vert < nNodesPerFace; ++face_vert) {
                const int vert = faces2verts[(face * nNodesPerFace) + face_vert];
                Omega_h::Vector<3> &coord = face_pts[face_vert];
                coord[0] = 0.5 * (oc(vert, 0) + nc(vert, 0));
                coord[1] = 0.5 * (oc(vert, 1) + nc(vert, 1));
                coord[2] = 0.5 * (oc(vert, 2) + nc(vert, 2));
            }
            Omega_h::Vector<3> areaNormal = Omega_h::cross(
                    face_pts[1] - face_pts[0], face_pts[2] - face_pts[0]);
            areaNormal[0] /= 2.;
            areaNormal[1] /= 2.;
            areaNormal[2] /= 2.;

            //make sure normal is outward
            const signed char code1 = face_use_codes[face_use];
            const bool        flip1 = Omega_h::code_is_flipped(code1);
            /*
        if (flip1) " normal points into element "
        else       " normal points out of element "
             */
            const int flipFactor = flip1 ? -1 : +1;
            areaNormal[0] *= flipFactor;
            areaNormal[1] *= flipFactor;
            areaNormal[2] *= flipFactor;

            Scalar faceFlux(0.0);
            int    nAttachedElements(0);
            for (int elem_use = faces2elem_uses[face];
                    elem_use < faces2elem_uses[face + 1]; ++elem_use) {
                const int elem2 = elem_uses2elems[elem_use];
                if (elem2 < 0) continue;
                ++nAttachedElements;
                for (int slot = 0; slot < 3; ++slot)
                    faceFlux += areaNormal[slot] * shockHeatFlux(elem2, slot);
            }
            faceFlux *= (nAttachedElements - 1);
            faceFlux /= 2.;

            const Scalar deltaE = -dt * faceFlux;
            internal_energy_per_unit_mass(elem1, state) +=
                    (deltaE / elem_mass(elem1));
            elem_energy(elem1) =
                    elem_mass(elem1) * internal_energy_per_unit_mass(elem1, state);

        }  //end for (int elem1_face = 0; elem1_face < nFacesPerElement; ++elem1_face)
    };

    Kokkos::parallel_for(mesh->nelems(), diffuse);

    return;
}  //end function apply()

template <int SpatialDim>
element_step<SpatialDim>::element_step(const Fields &mesh_fields)
: elem_node_connectivity(mesh_fields.femesh.elem_node_ids)
  , updatedCoordinates(Coordinates<Fields>())
  , elem_mass(ElementMass<Fields>())
  , elem_energy(ElementInternalEnergy<Fields>())
  , elem_volume(ElementVolume<Fields>())
  , planeWaveModulus(PlaneWaveModulus<Fields>())
  , element_force(ElementForce<Fields>())
  , F(DeformationGradient<Fields>())
  , vel_grad(VelocityGradient<Fields>())
  , mass_density(MassDensity<Fields>())
  , internalEnergy(InternalEnergyPerUnitMass<Fields>())
  , state1(0)
  , artificialViscosityModel(mesh_fields) {}

template <int SpatialDim>
void element_step<SpatialDim>::apply(const Fields &mesh_fields, const int arg_state1) {
    this->state1 = arg_state1;
    Kokkos::parallel_for(mesh_fields.femesh.nelems, *this);
}

template <int SpatialDim>
KOKKOS_INLINE_FUNCTION
void element_step<SpatialDim>::operator()(int ielem) const {
    Scalar x[ElemNodeCount], y[ElemNodeCount], z[ElemNodeCount];
    Scalar grad_x[ElemNodeCount], grad_y[ElemNodeCount], grad_z[ElemNodeCount];

    typename Fields::geom_array_type cgeom(
            Fields::getGeomFromSA(updatedCoordinates, state1));

    // Position:
    for (int i = 0; i < ElemNodeCount; ++i) {
        const int n = elem_node_connectivity(ielem, i);
        //use coordinates at next state
        x[i] = cgeom(n, 0);
        y[i] = cgeom(n, 1);
        if(SpatialDim == 3) z[i] = cgeom(n, 2);  //yuck.  hopefully this gets compiled out
    }

    // Gradient:
    comp_grad(x, y, z, grad_x, grad_y, grad_z);

    // lagrangian conservation of mass
    const Scalar vol = dot<ElemNodeCount>(x, grad_x);
    elem_volume(ielem) = vol;
    mass_density(ielem, state1) = elem_mass(ielem) / vol;
    elem_energy(ielem) = elem_mass(ielem) * internalEnergy(ielem, state1);
}

template <int SpatialDim>
assemble_forces<SpatialDim>::assemble_forces(const Fields &mesh_fields)
: node_elem_connectivity(mesh_fields.femesh.node_elem_ids)
  , nodal_mass(NodalMass<Fields>())
  , internal_force(InternalForce<Fields>())
  , element_force(ElementForce<Fields>()) {}

template <int SpatialDim>
void assemble_forces<SpatialDim>::apply(const Fields &mesh_fields) {
    assemble_forces op(mesh_fields);

    Kokkos::parallel_for(mesh_fields.femesh.nnodes, op);
}

template <int SpatialDim>
KOKKOS_INLINE_FUNCTION
void assemble_forces<SpatialDim>::operator()(int inode) const {
    // Getting count as per 'CSR-like' data structure
    const int begin = node_elem_connectivity.row_map[inode];
    const int end = node_elem_connectivity.row_map[inode + 1];

    Scalar local_force[] = {0.0, 0.0, 0.0};

    // Gather-sum internal force from
    // each element that a node is attached to.

    for (int i = begin; i < end; ++i) {
        //  node_elem_offset is a cumulative structure, so
        //  node_elem_offset(inode) should be the index where
        //  a particular row's elem_IDs begin
        const int nelem = node_elem_connectivity.entries(i, 0);

        //  find the row in an element's stiffness matrix
        //  that corresponds to inode
        const int elem_node_index = node_elem_connectivity.entries(i, 1);

        for(int d = 0; d < SpatialDim; ++d) {
            local_force[d] += element_force(nelem, d, elem_node_index);
        }
    }

    for(int d = 0; d < SpatialDim; ++d) {
        internal_force(inode, d) = local_force[d];
    }

}  //end operator()

template <int SpatialDim>
KOKKOS_INLINE_FUNCTION
ElementTallies<SpatialDim>::ElementTallies() {
    mass = 0.0;
    for(int d = 0; d < SpatialDim; ++d) {
        elementMomentum[d] = 0;
    }
    kineticEnergy = 0.0;
    internalEnergy = 0.0;
    magneticEnergy = 0.0;
}

template <int SpatialDim>
KOKKOS_INLINE_FUNCTION
void ElementTallies<SpatialDim>::operator+=(const volatile ElementTallies &addme) volatile {
    this->mass += addme.mass;

    for(int d = 0; d < SpatialDim; ++d) {
        this->elementMomentum[d] += addme.elementMomentum[d];
    }
    this->kineticEnergy += addme.kineticEnergy;
    this->internalEnergy += addme.internalEnergy;
    this->magneticEnergy += addme.magneticEnergy;
}

template <int SpatialDim>
GlobalTallies<SpatialDim>::GlobalTallies(const Fields &arg_mesh_fields, const int arg_state)
: numElements(arg_mesh_fields.femesh.nelems)
  , elem_node_connectivity(arg_mesh_fields.femesh.elem_node_ids)
  , updatedCoordinates(Coordinates<Fields>())
  , velocity(Fields::getGeomFromSA(Velocity<Fields>(), arg_state))
  , elem_mass(ElementMass<Fields>())
  , internalEnergy(InternalEnergyPerUnitMass<Fields>())
  , owned(arg_mesh_fields.femesh.omega_h_mesh->owned(SpatialDim))
  , state(arg_state) 
  , mhd(arg_mesh_fields) 
{
    int dataLength(0);
    dataLength += sizeof(ElementTallies<SpatialDim>) / sizeof(Scalar);
    dataLength += 1;
    contiguousMemoryTallies.clear();
    contiguousMemoryTallies.resize(dataLength);
}

template<int SpatialDim>
KOKKOS_INLINE_FUNCTION 
Scalar 
GlobalTallies<SpatialDim>::elementMagneticEnergy( const int ielem ) const
{  
  typename Fields::geom_array_type cgeom(Fields::getGeomFromSA(updatedCoordinates, state));
  constexpr int ElemNodeCount = Fields::ElemNodeCount;
  Scalar x[ElemNodeCount], y[ElemNodeCount], z[ElemNodeCount];
  for (int i = 0; i < ElemNodeCount; ++i) {
    const int n = elem_node_connectivity(ielem, i);
    x[i] = cgeom(n, 0);
    y[i] = cgeom(n, 1);
    z[i] = cgeom(n, 2);
  }
  return mhd.elementMagneticEnergy(ielem, x,y,z);
}

template <int SpatialDim>
OMEGA_H_DEVICE
void GlobalTallies<SpatialDim>::operator()(const int ielem, ElementTallies<SpatialDim> &localSum) const {
    if (!owned[ielem]) return;
    const int ElemNodeCount = Fields::ElemNodeCount;

    const Scalar emass = elem_mass(ielem);

    localSum.mass += emass;

    const Scalar nodeMass = emass / ElemNodeCount;
    for (int nodeCount = 0; nodeCount < ElemNodeCount; ++nodeCount) {
        const int inode = elem_node_connectivity(ielem, nodeCount);

        for (int slot = 0; slot < SpatialDim; ++slot) {
            const Scalar v = velocity(inode, slot);
            const Scalar mom = nodeMass * v;
            localSum.elementMomentum[slot] += mom;
            localSum.kineticEnergy += 0.5 * mom * v;
        }  //end for (int slot=0; slot<3; ++slot)

    }  //end for (int i=0; i<ElemNodeCount; ++i)

    localSum.internalEnergy += emass * internalEnergy(ielem, state);
    
    if (SpatialDim>2)
      localSum.magneticEnergy += this->elementMagneticEnergy(ielem);
}

template <int SpatialDim>
void GlobalTallies<SpatialDim>::apply() {
    ElementTallies<SpatialDim> elementSum;
    Kokkos::parallel_reduce(numElements, *this, elementSum);
    const Scalar totalEnergy = elementSum.internalEnergy + elementSum.kineticEnergy + elementSum.magneticEnergy;

    contiguousMemoryTallies[0] = elementSum.mass;

    for(int d = 1; d <= SpatialDim; ++d)
        contiguousMemoryTallies[d] = elementSum.elementMomentum[d-1];

    contiguousMemoryTallies[SpatialDim + 1] = elementSum.kineticEnergy;
    contiguousMemoryTallies[SpatialDim + 2] = elementSum.internalEnergy;
    contiguousMemoryTallies[SpatialDim + 3] = elementSum.magneticEnergy;
    contiguousMemoryTallies[SpatialDim + 4] = totalEnergy;
}

template <int SpatialDim>
Scalar get_min_density(
        comm::Machine const &                                   machine,
        typename Fields<SpatialDim>::array_type densities) {
    auto nelems = int(densities.size());
    auto f = LAMBDA_EXPRESSION(int ielem, Scalar &update) {
        update = Omega_h::min2(densities(ielem), update);
    };
    Scalar result;
    auto   reducer =
            Kokkos::Min<Scalar, Kokkos::DefaultExecutionSpace>(result);
    Kokkos::parallel_reduce(nelems, f, reducer);
    return comm::min(machine, result);
}

template <int SpatialDim>
void check_densities(
        comm::Machine const &                     machine,
        Fields<SpatialDim> const &mesh_fields,
        int                                       state,
        Scalar                                    min_mass_density_allowed,
        Scalar                                    min_energy_density_allowed) {
    typedef lgr::Fields<SpatialDim> Fields;
    auto mass_density = mesh_fields.getFromSA(MassDensity<Fields>(), state);
    auto energy_density = InternalEnergyDensity<Fields>();
    auto min_mass_density =
            get_min_density<SpatialDim>(machine, mass_density);
    auto min_energy_density =
            get_min_density<SpatialDim>(machine, energy_density);
    if (!comm::rank(machine)) {
        if (min_mass_density < min_mass_density_allowed) {
            std::cout << "WARNING: MINIMUM MASS DENSITY " << min_mass_density
                    << " LESS THAN ALLOWED " << min_mass_density_allowed << '\n';
        }
        if (min_energy_density < min_energy_density_allowed) {
            std::cout << "WARNING: INTERNAL ENERGY DENSITY " << min_mass_density
                    << " LESS THAN ALLOWED " << min_mass_density_allowed << '\n';
        }
    }
}

#define LGR_EXPL_INST(SpatialDim) \
    template struct ArtificialViscosity<SpatialDim>; \
    template class explicit_time_step<SpatialDim>; \
    template struct initialize_element<SpatialDim>; \
    template struct initialize_node<SpatialDim>; \
    template struct update_node_mass_after_remap<SpatialDim>; \
    template struct initialize_time_step_elements<SpatialDim>; \
    template struct grad<SpatialDim>; \
    template struct GRAD<SpatialDim>; \
    template struct internal_force<SpatialDim>; \
    template struct energy_step<SpatialDim>; \
    template struct shock_heat_flux_step<SpatialDim>; \
    template struct element_step<SpatialDim>; \
    template struct assemble_forces<SpatialDim>; \
    template struct GlobalTallies<SpatialDim>; \
    template struct ElementTallies<SpatialDim>; \
    template Scalar get_min_density<SpatialDim>( \
                                                 comm::Machine const &                                   machine, \
                                                 typename Fields<SpatialDim>::array_type densities); \
                                                 template void check_densities( \
                                                                                comm::Machine const &                     machine, \
                                                                                Fields<SpatialDim> const &mesh_fields, \
                                                                                int    state, \
                                                                                Scalar min_mass_density_allowed, \
                                                                                Scalar min_energy_density_allowed);
LGR_EXPL_INST(3)
LGR_EXPL_INST(2)
#undef LGR_EXPL_INST

template struct initialize_node<1>;

} /* namespace lgr */
