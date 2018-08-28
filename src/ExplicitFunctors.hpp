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

#ifndef LGR_EXPLICIT_FUNCTORS_HPP
#define LGR_EXPLICIT_FUNCTORS_HPP

#include "MagnetoHydroDynamics.hpp"
#include "InitialConditions.hpp"

namespace lgr {

template <int SpatialDim>
struct ArtificialViscosity {
    typedef ExecSpace execution_space;
    typedef lgr::Fields<SpatialDim> Fields;

    Scalar                                  lin_bulk_visc;
    Scalar                                  quad_bulk_visc;
    const typename Fields::array_type       elem_volume;
    const typename Fields::array_type       elem_mass;
    const typename Fields::elem_tensor_type vel_grad;
    const typename Fields::state_array_type planeWaveModulus;

    const MHD<SpatialDim> mhd;

    ArtificialViscosity(const Fields &meshFields);

    KOKKOS_INLINE_FUNCTION
    Omega_h::Few<Scalar, Fields::SymTensorLength> kineticViscosity(
            int           ielem,
            int           state0,
            int           state1,
            const Scalar *x,
            const Scalar *y,
            const Scalar *z ) const;

    KOKKOS_INLINE_FUNCTION
    Omega_h::Few<Scalar, SpatialDim> heatFlux(
            int           ielem,
            int           state0,
            int           state1,
            const Scalar *x,
            const Scalar *y,
            const Scalar *z,
            const Scalar *grad_x,
            const Scalar *grad_y,
            const Scalar *grad_z,
            const Scalar *nodalPressure) const;
};

template <int SpatialDim>
class explicit_time_step {
public:
    typedef ExecSpace                                         execution_space;
    typedef lgr::Fields<SpatialDim> Fields;
    typedef Scalar                                             value_type;

public:
    Scalar lin_bulk_visc;
    Scalar quad_bulk_visc;
    Scalar c_tau;

    typename Fields::elem_node_ids_type          elem_node_connectivity;
    const typename Fields::geom_state_array_type updatedCoordinates;
    const typename Fields::array_type            elem_volume;
    const typename Fields::array_type            elem_mass;
    const typename Fields::array_type            element_time_step;
    const typename Fields::elem_tensor_type      vel_grad;
    const typename Fields::state_array_type      planeWaveModulus;
    const typename Fields::array_type            elem_time_step;

    const int numElements;
    int       state;

    const MHD<SpatialDim> mhd;

    explicit_time_step(const Fields &meshFields);

    KOKKOS_INLINE_FUNCTION
    void init(value_type &update) const;

    KOKKOS_INLINE_FUNCTION
    void join(
            volatile value_type &update, const volatile value_type &source) const;

    KOKKOS_INLINE_FUNCTION
    Scalar elementTimeStep(
            int           ielem,
            const Scalar *x,
            const Scalar *y,
            const Scalar *z,
            const Scalar *grad_x,
            const Scalar *grad_y,
            const Scalar *grad_z) const;

    KOKKOS_INLINE_FUNCTION
    void operator()(int ielem, value_type &update) const;

    Scalar apply(const int arg_state);
};

template <int SpatialDim>
struct initialize_element {
    typedef ExecSpace execution_space;

    typedef lgr::Fields<SpatialDim> Fields;

    typename Fields::elem_node_ids_type elem_node_connectivity;
    typename Fields::elem_tensor_type   F;
    typename Fields::elem_tensor_type   Fold;
    typename Fields::array_type         elem_mass;
    typename Fields::array_type         elem_energy;
    typename Fields::array_type         elem_volume;
    typename Fields::state_array_type   internal_energy_per_unit_mass;
    typename Fields::array_type         internal_energy_density;
    typename Fields::state_array_type   planeWaveModulus;
    typename Fields::state_array_type   mass_density;
    typename Fields::node_coords_type   model_coords;

    static const int ElemNodeCount = Fields::ElemNodeCount;

    initialize_element(
            const Fields &mesh_fields, const InitialConditions<Fields> &ic);

    KOKKOS_INLINE_FUNCTION
    void operator()(int ielem) const;

    static void apply(
            const Fields &mesh_fields, const InitialConditions<Fields> &ic);
};

template <int SpatialDim>
struct initialize_node {
    typedef ExecSpace execution_space;

    typedef lgr::Fields<SpatialDim> Fields;

    typename Fields::node_elem_ids_type node_elem_connectivity;
    typename Fields::array_type         nodal_mass;
    typename Fields::array_type         elem_mass;
    typename Fields::node_coords_type   model_coords;
    typename Fields::geom_array_type    coord_subview_state_0;
    typename Fields::geom_array_type    coord_subview_state_1;

    static const int ElemNodeCount = Fields::ElemNodeCount;

    initialize_node(const Fields &mesh_fields);

    KOKKOS_INLINE_FUNCTION
    void operator()(int inode) const;

    static void apply(const Fields &mesh_fields);
};

template <int SpatialDim>
struct update_node_mass_after_remap {
    typedef ExecSpace execution_space;

    typedef lgr::Fields<SpatialDim> Fields;

    typename Fields::node_elem_ids_type node_elem_connectivity;
    typename Fields::array_type         nodal_mass;
    typename Fields::array_type         elem_mass;
    typename Fields::geom_array_type    velocity;
    int                                 state;

    static const int ElemNodeCount = Fields::ElemNodeCount;

    update_node_mass_after_remap(const Fields &mesh_fields, int arg_state);

    KOKKOS_INLINE_FUNCTION
    void operator()(int inode) const;

    static void apply(const Fields &mesh_fields, int arg_state);
};

template <int SpatialDim>
struct initialize_time_step_elements {
    typedef ExecSpace                          execution_space;
    typedef typename execution_space::size_type size_type;

    typedef lgr::Fields<SpatialDim> Fields;

    const typename Fields::state_array_type mass_density;
    const typename Fields::state_array_type internal_energy_per_unit_mass;
    const typename Fields::state_array_type planeWaveModulus;
    const typename Fields::state_array_type bulkModulus;
    const typename Fields::elem_sym_tensor_state_type stress;
    typename Fields::elem_tensor_type                 F;
    typename Fields::elem_tensor_type                 Fold;
    const typename Fields::elem_vector_state_type     uprime;

    const int state0;
    const int state1;

    initialize_time_step_elements(
            const Fields &mesh_fields, const int arg_state0, const int arg_state1);

    static void apply(
            const Fields &mesh_fields, const int arg_state0, const int arg_state1);

    KOKKOS_INLINE_FUNCTION
    void operator()(int ielem) const;
};

template <int SpatialDim>
struct grad {
    typedef ExecSpace execution_space;

    typedef lgr::Fields<SpatialDim> Fields;

    static const int ElemNodeCount = Fields::ElemNodeCount;

    // Global arrays used by this functor.

    const typename Fields::elem_node_ids_type elem_node_connectivity;
    typename Fields::geom_array_type          velocity[2];
    typename Fields::geom_array_type          xn;
    typename Fields::geom_array_type          xnp1;
    const typename Fields::elem_tensor_type   vel_grad;
    const typename Fields::array_type         elem_volume;

    int    state0;
    int    state1;
    Scalar alpha;

    // Constructor on the Host to populate this device functor.
    // All array view copies are shallow.
    grad(
            const Fields &fields,
            const int     arg_state0,
            const int     arg_state1,
            const Scalar  arg_alpha);

    //   Calculate Velocity Gradients
    KOKKOS_INLINE_FUNCTION
    void v_grad(
            int     ielem,
            Scalar *vx,
            Scalar *vy,
            Scalar *vz,
            Scalar *grad_x,
            Scalar *grad_y,
            Scalar *grad_z,
            Scalar  inv_vol) const;

    KOKKOS_INLINE_FUNCTION
    void operator()(int ielem) const;

    static void apply(
            const Fields &fields,
            const int     arg_state0,
            const int     arg_state1,
            const Scalar  arg_alpha);
};

template <int SpatialDim>
struct GRAD {
    typedef ExecSpace execution_space;

    typedef lgr::Fields<SpatialDim> Fields;

    static const int ElemNodeCount = Fields::ElemNodeCount;

    // Global arrays used by this functor.

    const typename Fields::elem_node_ids_type elem_node_connectivity;
    typename Fields::geom_array_type          xn;
    typename Fields::geom_array_type          xnp1;
    const typename Fields::elem_tensor_type   F;
    const typename Fields::elem_tensor_type   Fold;

    int    state0;
    int    state1;
    Scalar alpha;

    // Constructor on the Host to populate this device functor.
    // All array view copies are shallow.
    GRAD(
            const Fields &fields,
            const int     arg_state0,
            const int     arg_state1,
            const Scalar  arg_alpha);

    //   Calculate deformation gradient
    KOKKOS_INLINE_FUNCTION
    void deformationGradient(
            int     ielem,
            Scalar *x,
            Scalar *y,
            Scalar *z,
            Scalar *GRAD_X,
            Scalar *GRAD_Y,
            Scalar *GRAD_Z,
            Scalar  inv_vol) const;

    //--------------------------------------------------------------------------
    // Functor operator() which calls the three member functions.

    KOKKOS_INLINE_FUNCTION
    void operator()(int ielem) const;

    static void apply(
            const Fields &fields,
            const int     arg_state0,
            const int     arg_state1,
            const Scalar  arg_alpha);
};

//-----------------------------------------------------------------------------------

template <int SpatialDim>
struct internal_force {
    typedef ExecSpace execution_space;

    typedef lgr::Fields<SpatialDim> Fields;

    typedef typename Fields::size_type size_type;

    static const int ElemNodeCount = Fields::ElemNodeCount;


    // Global arrays used by this functor.

    const typename Fields::elem_node_ids_type elem_node_connectivity;
    const typename Fields::geom_state_array_type      updatedCoordinates;
    const typename Fields::array_type                 elem_mass;
    const typename Fields::array_type                 elem_volume;
    const typename Fields::elem_sym_tensor_state_type stress;
    const typename Fields::elem_node_geom_type        element_force;
    const typename Fields::elem_tensor_type           vel_grad;
    const typename Fields::array_type                 pprime;
    const typename Fields::array_type                 nodal_pressure;
    typename Fields::geom_array_type                  velocity[2];

    const int state0;
    const int state1;

    const ArtificialViscosity<SpatialDim> artificialViscosityModel;

    const MHD<SpatialDim> mhd;

    internal_force(const Fields &mesh_fields, const int arg_state0, const int arg_state1);

    static void apply(const Fields &mesh_fields, const int arg_state0, const int arg_state1);

    KOKKOS_INLINE_FUNCTION
    void comp_force( int ielem,
		     const Scalar *const grad_x,
		     const Scalar *const grad_y,
		     const Scalar *const grad_z,
		     Scalar *algoStress) const;

    KOKKOS_INLINE_FUNCTION
    void operator()(int ielem) const;
};

template <int SpatialDim>
struct energy_step {
    typedef ExecSpace execution_space;

    typedef lgr::Fields<SpatialDim> Fields;

    static const int ElemNodeCount = Fields::ElemNodeCount;

    // Global arrays used by this functor.

    const typename Fields::elem_node_ids_type    elem_node_connectivity;
    const typename Fields::geom_state_array_type updatedCoordinates;
    typename Fields::geom_array_type             velocity[2];
    const typename Fields::array_type            elem_mass;
    const typename Fields::array_type            elem_energy;
    const typename Fields::array_type            elem_volume;
    const typename Fields::state_array_type      internal_energy_per_unit_mass;
    const typename Fields::array_type            internal_energy_density;
    const typename Fields::elem_sym_tensor_state_type stress;
    const typename Fields::elem_node_geom_type        element_force;
    const typename Fields::elem_tensor_type           vel_grad;
    const typename Fields::array_type                 pprime;
    const typename Fields::array_type                 nodal_pressure;
    const typename Fields::elem_vector_state_type     uprime;
    const typename Fields::elem_vector_type           shockHeatFlux;

    const Scalar dt_vel;
    const int    state0;
    const int    state1;

    const ArtificialViscosity<SpatialDim>
    artificialViscosityModel;

    energy_step(
            const Fields &mesh_fields,
            const Scalar  arg_dt,
            const int     arg_state0,
            const int     arg_state1);

    static void apply(
            const Fields &mesh_fields,
            const Scalar  arg_dt,
            const int     arg_state0,
            const int     arg_state1);

    KOKKOS_INLINE_FUNCTION
    Scalar comp_work(
            int                 ielem,
            const Scalar *      vx,
            const Scalar *      vy,
            const Scalar *      vz,
            const Scalar *const grad_x,
            const Scalar *const grad_y,
            const Scalar *const grad_z,
            const Scalar *      algoStress) const;

    KOKKOS_INLINE_FUNCTION
    void operator()(int ielem) const;
};

template <int SpatialDim>
struct shock_heat_flux_step {
    typedef lgr::Fields<SpatialDim> Fields;

    void apply(
            Omega_h::Mesh *mesh,
            Fields &       mesh_fields,
            const int      state,
            const Scalar   dt);

};  //end struct shock_heat_flux_step

template <int SpatialDim>
struct element_step {
    typedef lgr::Fields<SpatialDim> Fields;

    static constexpr int ElemNodeCount = Fields::ElemNodeCount;

    typedef Scalar value_type;

    // Global arrays used by this functor.

    const typename Fields::elem_node_ids_type    elem_node_connectivity;
    const typename Fields::geom_state_array_type updatedCoordinates;
    const typename Fields::array_type            elem_mass;
    const typename Fields::array_type            elem_energy;
    const typename Fields::array_type            elem_volume;
    const typename Fields::state_array_type      planeWaveModulus;
    const typename Fields::elem_node_geom_type   element_force;
    const typename Fields::elem_tensor_type      F;
    const typename Fields::elem_tensor_type      vel_grad;
    const typename Fields::state_array_type      mass_density;
    const typename Fields::state_array_type      internalEnergy;

    int state1;

    const ArtificialViscosity<SpatialDim>
    artificialViscosityModel;

    element_step(const Fields &mesh_fields);

    void apply(const Fields &mesh_fields, const int arg_state1);

    KOKKOS_INLINE_FUNCTION
    void operator()(int ielem) const;
};

template <int SpatialDim>
struct assemble_forces {
    typedef ExecSpace                          execution_space;
    typedef typename execution_space::size_type size_type;

    typedef lgr::Fields<SpatialDim> Fields;

    const typename Fields::node_elem_ids_type  node_elem_connectivity;
    const typename Fields::array_type          nodal_mass;
    const typename Fields::geom_array_type     internal_force;
    const typename Fields::elem_node_geom_type element_force;

    assemble_forces(const Fields &mesh_fields);

    static void apply(const Fields &mesh_fields);

    KOKKOS_INLINE_FUNCTION
    void operator()(int inode) const;
};

template <int SpatialDim>
struct ElementTallies {
    Scalar mass;
    Scalar elementMomentum[3];
    Scalar kineticEnergy;
    Scalar internalEnergy;
    Scalar magneticEnergy;

    KOKKOS_INLINE_FUNCTION
    ElementTallies();

    KOKKOS_INLINE_FUNCTION
    void operator+=(const volatile ElementTallies &addme) volatile;
};

template <int SpatialDim>
struct GlobalTallies {
    typedef ExecSpace                                         execution_space;
    typedef lgr::Fields<SpatialDim> Fields;
    typedef typename Fields::size_type size_type;

    const int                                 numElements;
    const typename Fields::elem_node_ids_type elem_node_connectivity;
    const typename Fields::geom_state_array_type updatedCoordinates;
    const typename Fields::geom_array_type    velocity;
    const typename Fields::array_type         elem_mass;
    const typename Fields::state_array_type   internalEnergy;

    Omega_h::Bytes                            owned;

    const int state;

    const MHD<SpatialDim> mhd;

    //global tallies to save/output
    std::vector<double> contiguousMemoryTallies;

    GlobalTallies(const Fields &arg_mesh_fields, const int arg_state);

    KOKKOS_INLINE_FUNCTION 
    Scalar 
    elementMagneticEnergy( const int ielem ) const;
  
    OMEGA_H_DEVICE
    void operator()(const int ielem, ElementTallies<SpatialDim> &localSum) const;

    void apply();

};  //end struct GlobalTallies

template <int SpatialDim>
Scalar get_min_density(
        comm::Machine const &                                   machine,
        typename Fields<SpatialDim>::array_type densities);

template <int SpatialDim>
void check_densities(
        comm::Machine const &                     machine,
        Fields<SpatialDim> const &mesh_fields,
        int                                       state,
        Scalar                                    min_mass_density_allowed,
        Scalar                                    min_energy_density_allowed);

#define LGR_EXPL_INST_DECL(SpatialDim) \
    extern template struct ArtificialViscosity<SpatialDim>; \
    extern template class explicit_time_step<SpatialDim>; \
    extern template struct initialize_element<SpatialDim>; \
    extern template struct initialize_node<SpatialDim>; \
    extern template struct update_node_mass_after_remap<SpatialDim>; \
    extern template struct initialize_time_step_elements<SpatialDim>; \
    extern template struct grad<SpatialDim>; \
    extern template struct GRAD<SpatialDim>; \
    extern template struct internal_force<SpatialDim>; \
    extern template struct energy_step<SpatialDim>; \
    extern template struct shock_heat_flux_step<SpatialDim>; \
    extern template struct element_step<SpatialDim>; \
    extern template struct assemble_forces<SpatialDim>; \
    extern template struct GlobalTallies<SpatialDim>; \
    extern template struct ElementTallies<SpatialDim>; \
    extern template Scalar get_min_density<SpatialDim>( \
                                                        comm::Machine const &                                   machine, \
                                                        typename Fields<SpatialDim>::array_type densities); \
                                                        extern template void check_densities( \
                                                                                              comm::Machine const &                     machine, \
                                                                                              Fields<SpatialDim> const &mesh_fields, \
                                                                                              int    state, \
                                                                                              Scalar min_mass_density_allowed, \
                                                                                              Scalar min_energy_density_allowed);
//LGR_EXPL_INST_DECL(3)
//LGR_EXPL_INST_DECL(2)
#undef LGR_EXPL_INST_DECL

} /* namespace lgr */

#endif
