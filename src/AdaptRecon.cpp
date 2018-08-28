#include "AdaptRecon.hpp"

#include <Omega_h_array_ops.hpp>
#include <Omega_h_functors.hpp>
#include <Omega_h_metric.hpp>
#include <Omega_h_quality.hpp>
#include <Omega_h_shape.hpp>
#include <Omega_h_teuchos.hpp>
#include "ExplicitLambdas.hpp"
#include "ExplicitFunctors.hpp"
#include "FieldDB.hpp"
#include "Fields.hpp"

#include "LGRLambda.hpp"

#ifndef OMEGA_H_USE_MPI
#error "rebuild Omega_h with -DOmega_h_USE_MPI:BOOL=ON"
#endif

#if !(defined(OMEGA_H_USE_KOKKOSCORE) && defined(OMEGA_H_USE_TEUCHOS))
#error "rebuild Omega_h with -DOmega_h_USE_Trilinos:BOOL=ON"
#endif

namespace lgr {

/* assigns to each element the integral of momentum in it */
template <int SpatialDim>
void momentum_to_elements(
    Fields<SpatialDim>& fields, int state) {
  using DefaultFields = Fields<SpatialDim>;
  auto elems2nodes = fields.femesh.elem_node_ids;
  auto elems2mass = ElementMass<DefaultFields>();
  auto nodes2velocity = fields.getGeomFromSA(Velocity<DefaultFields>(), state);
  auto elems2momentum =
      FieldDB<typename DefaultFields::geom_array_type>::Self().at(
          "element momentum");
  auto f = LAMBDA_EXPRESSION(int elem) {
    auto          mass = elems2mass(elem);
    Scalar avg_velocity[SpatialDim] = {};
    for (int i = 0; i < DefaultFields::ElemNodeCount; ++i) {
      auto node = elems2nodes(elem, i);
      for (int j = 0; j < SpatialDim; ++j) {
        avg_velocity[j] += nodes2velocity(node, j);
      }
    }
    for (int j = 0; j < SpatialDim; ++j)
      avg_velocity[j] /= DefaultFields::ElemNodeCount;
    for (int j = 0; j < SpatialDim; ++j)
      elems2momentum(elem, j) = avg_velocity[j] * mass;
  };
  Kokkos::parallel_for(fields.femesh.nelems, f);
}

template <int SpatialDim>
struct ElemMomentumSum : public Omega_h::SumFunctor<Scalar> {
  using DefaultFields = Fields<SpatialDim>;
  int dim;
  using array_type = typename DefaultFields::geom_array_type;
  array_type     elems2momentum;
  Omega_h::Bytes owned;
  ElemMomentumSum(int dim_, array_type elems2momentum_, Omega_h::Bytes owned_)
      : dim(dim_), elems2momentum(elems2momentum_), owned(owned_) {}
  KOKKOS_INLINE_FUNCTION
  void operator()(const size_t& elem, Scalar& update) const {
    if (owned[int(elem)]) {
      update += elems2momentum(elem, dim);
    }
  }
};

template <int SpatialDim>
Omega_h::Vector<SpatialDim> integrate_momentum(
    comm::Machine                                    machine,
    Fields<SpatialDim>& fields) {
  using DefaultFields = Fields<SpatialDim>;
  Omega_h::Vector<SpatialDim> integral;
  auto owned = fields.femesh.omega_h_mesh->owned(SpatialDim);
  auto elems2momentum =
      FieldDB<typename DefaultFields::geom_array_type>::Self().at(
          "element momentum");
  for (int dim = 0; dim < SpatialDim; ++dim) {
    Scalar local_sum = 0;
    auto          f = ElemMomentumSum<SpatialDim>(dim, elems2momentum, owned);
    Kokkos::parallel_reduce(fields.femesh.nelems, f, local_sum);
    auto global_sum = comm::sum(machine, local_sum);
    integral[dim] = global_sum;
  }
  return integral;
}

template <int SpatialDim>
void debug_momentum(
    comm::Machine                                    machine,
    Fields<SpatialDim>& fields) {
  auto integral = integrate_momentum(machine, fields);
  if (!comm::rank(machine)) {
    std::cout << "momentum integral:" << ' ' << integral[0] << ' '
              << integral[1] << ' ' << integral[2] << '\n';
  }
}

template <class ArrayType>
struct ArrayMax : public Omega_h::MaxFunctor<typename ArrayType::value_type> {
  ArrayType a_;
  ArrayMax(ArrayType a) : a_(a) {}
  KOKKOS_INLINE_FUNCTION void operator()(size_t i, double& update) const {
    update = Omega_h::max2(update, a_(i));
  }
};

template <int SpatialDim>
AdaptRecon<SpatialDim>::AdaptRecon(
    Teuchos::ParameterList& problem,
    DefaultFields&          mesh_fields_,
    comm::Machine           machine_)
    : is_adaptive(false)
    , always_adapt(false)
    , trigger_quality(0.28)
    , trigger_length(2.2)
    , should_debug_momentum(false)
    , should_advect_original_implied(false)
    , mesh_fields(mesh_fields_)
    , machine(machine_)
    , adapt_opts(3)
    , metric_input() {
  auto m = mesh_fields.femesh.omega_h_mesh;
  if (problem.isSublist("Adaptivity")) {
    // presence of the sublist means that adaptation should be activated
    is_adaptive = true;
    auto& adapt_pl = problem.sublist("Adaptivity");
    if (adapt_pl.isSublist("Omega_h")) {
      auto& oshPL = adapt_pl.sublist("Omega_h");
      Omega_h::update_adapt_opts(&adapt_opts, oshPL);
      if (oshPL.isSublist("Metric")) {
        auto& metricPL = oshPL.sublist("Metric");
        Omega_h::update_metric_input(&metric_input, metricPL);
      }
    }
    always_adapt = true;
    Omega_h::set_if_given(&always_adapt, adapt_pl, "Every Step");
    Omega_h::set_if_given(&trigger_quality, adapt_pl, "Trigger Quality");
    Omega_h::set_if_given(&trigger_length, adapt_pl, "Trigger Length");
    Omega_h::set_if_given(&should_debug_momentum, adapt_pl, "Debug Momentum");
    Omega_h::set_if_given(
        &should_advect_original_implied, adapt_pl, "Advect Original Implied");
    always_recompute_metric = false;
    Omega_h::set_if_given(
        &always_recompute_metric, adapt_pl, "Always Recompute Metric");
    if (adapt_pl.isSublist("Metric Tags")) {
      auto& metric_tags_pl = adapt_pl.sublist("Metric Tags");
      Omega_h::update_tag_set(&metric_tags, m->dim(), metric_tags_pl);
    }
    if (adapt_pl.isSublist("Adapt Tags")) {
      auto& adapt_tags_pl = adapt_pl.sublist("Adapt Tags");
      Omega_h::update_tag_set(&adapt_tags, m->dim(), adapt_tags_pl);
    } else {
      auto dim = size_t(m->dim());
      adapt_tags[0].insert("vel");
      adapt_tags[dim].insert("mass_density");
      adapt_tags[dim].insert("internal_energy_density");
      adapt_tags[dim].insert("deformation_gradient");
      adapt_tags[dim].insert("fine_scale_displacement");
    }
    if (!m->has_tag(0, "original_implied") && should_advect_original_implied) {
      // Sample the original implied metric
      auto msh = mesh_fields.femesh.omega_h_mesh;
      auto metrics = Omega_h::get_implied_isos(msh);
      msh->add_tag(0, "original_implied", 1, metrics);
    }
  }
  if (problem.isSublist("Restart") &&
      problem.sublist("Restart").isSublist("Tags")) {
    auto& restart_pl = problem.sublist("Restart");
    auto& restart_tags_pl = restart_pl.sublist("Tags");
    Omega_h::update_tag_set(&restart_tags, m->dim(), restart_tags_pl);
  } else {
    auto dim = size_t(m->dim());
    restart_tags[0].insert("coordinates");
    restart_tags[0].insert("velocity");
    restart_tags[dim].insert("mass_density");
    restart_tags[dim].insert("internal_energy_density");
    restart_tags[dim].insert("deformation_gradient");
    restart_tags[dim].insert("fine_scale_displacement");
    // When we adapt, we recompute densities based on amounts per element,
    // but if we try to do this for restart files the densities change
    // ever so slightly due to the wonderful thing that is floating-point math.
    // therefore, for restart files, we store these somewhat redundant values
    // in order to get exactly the same results when restarting
    restart_tags[dim].insert("mass");
    restart_tags[dim].insert("internal_energy");
    restart_tags[dim].insert("internal_energy_per_mass");
  }
}

template <int SpatialDim>
void AdaptRecon<SpatialDim>::preAdapt(int next_state) {
  // Move any fields that need to be remapped during adaptation to the mesh
  mesh_fields.copyTagsToMesh(adapt_tags, next_state);
  if (should_debug_momentum) {
    momentum_to_elements(mesh_fields, next_state);
    if (!comm::rank(machine)) {
      std::cout << "before adapt: ";
    }
    debug_momentum(machine, mesh_fields);
  }
}

template <int SpatialDim>
void AdaptRecon<SpatialDim>::postAdapt(MeshIO& out_obj, int next_state) {
  mesh_fields.resize();
  mesh_fields.copyCoordsFromMesh(next_state);
  mesh_fields.copyTagsFromMesh(adapt_tags, next_state);
  mesh_fields.cleanTagsFromMesh(adapt_tags);
  out_obj.computeSets();
  /*
    update element volume.
    update velocity gradient (needed for time step calculations).
    nodal velocity must be post-remap correct before this call to grad.
  */
  const Scalar alpha(1.0);
  grad<SpatialDim>::apply(
      mesh_fields, next_state, next_state, alpha);

  //update elements after remap (assume deformation gradient F has been re-mapped)
  update_elements_after_remap<SpatialDim>(mesh_fields, next_state);

  update_node_mass_after_remap<SpatialDim>::apply(
      mesh_fields, next_state);
  //re-lump mass matrix
  mesh_fields.conform("nodal_mass", NodalMass<DefaultFields>());
  if (should_debug_momentum) {
    momentum_to_elements(mesh_fields, next_state);
    if (!comm::rank(machine)) {
      std::cout << "after adapt: ";
    }
    debug_momentum(machine, mesh_fields);
  }
}

template <int SpatialDim>
bool AdaptRecon<SpatialDim>::shouldAdapt(Omega_h::AdaptOpts const& opts) const {
  if (always_adapt) return true;
  auto   m = mesh_fields.femesh.omega_h_mesh;
  double minqual, maxlen;
  if (m->has_tag(0, "target_metric")) {
    auto quals = Omega_h::measure_qualities(
        m, m->template get_array<double>(0, "target_metric"));
    auto lens = Omega_h::measure_edges_metric(
        m, m->template get_array<double>(0, "target_metric"));
    minqual = Omega_h::get_min(m->comm(), quals);
    maxlen = Omega_h::get_max(m->comm(), lens);
  } else {
    /* this wrapper is for the allow-pinching system. */
    minqual = Omega_h::min_fixable_quality(m, opts);
    maxlen = m->max_length();
  }
  if (minqual < trigger_quality) return true;
  if (maxlen > trigger_length) return true;
  return false;
}

template <int SpatialDim>
void AdaptRecon<SpatialDim>::writeRestart(
    std::string const& path, const int next_state) {
  auto m = mesh_fields.femesh.omega_h_mesh;
  mesh_fields.copyTagsToMesh(restart_tags, next_state);
  Omega_h::binary::write(path, m);
  mesh_fields.cleanTagsFromMesh(restart_tags);
}

template <int SpatialDim>
void AdaptRecon<SpatialDim>::loadRestart(
    MeshIO& out_obj) {
  auto& f = mesh_fields;
  f.resize();
  f.copyTagsFromMesh(restart_tags, 0);
  f.copyTagsFromMesh(restart_tags, 1);
  f.cleanTagsFromMesh(restart_tags);
  out_obj.computeSets();
  initialize_node<SpatialDim>::apply(mesh_fields);
}

template <int SpatialDim>
void AdaptRecon<SpatialDim>::computeMetric( const int /*current_state*/, 
					    const int next_state) {
  mesh_fields.copyTagsToMesh(metric_tags, next_state);
  auto m = mesh_fields.femesh.omega_h_mesh;
  m->remove_tag(0, "target_metric");
  Omega_h::generate_target_metric_tag(m, metric_input);
  mesh_fields.cleanTagsFromMesh(metric_tags);
  if (!m->has_tag(0, "metric")) {
    Omega_h::add_implied_metric_based_on_target(m);
  }
}

template <int SpatialDim>
bool AdaptRecon<SpatialDim>::adaptMeshAndRemapFields(
    MeshIO& out_obj, const int current_state, const int next_state) {
  if (!isAdaptive()) return false;
  /* regardless of what the metric logic needs, we must have
   * new coordinates in order to properly measure quality and length
   * in shouldAdapt() */
  mesh_fields.copyCoordsToMesh(next_state);
  if (always_recompute_metric) {
    computeMetric(current_state, next_state);
  }
  if (!shouldAdapt(adapt_opts)) return false;
  if (!always_recompute_metric) {
    computeMetric(current_state, next_state);
  }
  auto m = mesh_fields.femesh.omega_h_mesh;
  preAdapt(next_state);
  while (Omega_h::approach_metric(m, adapt_opts)) {
    Omega_h::adapt(m, adapt_opts);
  }
  m->set_parting(OMEGA_H_GHOSTED);
  postAdapt(out_obj, next_state);
  return true;
}

// explicit instantiation:
template class AdaptRecon<3>;
template class AdaptRecon<2>;

}  // namespace lgr
