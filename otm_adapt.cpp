#include <hpc_array.hpp>
#include <hpc_array_vector.hpp>
#include <hpc_dimensional.hpp>
#include <hpc_execution.hpp>
#include <hpc_numeric.hpp>
#include <hpc_quaternion.hpp>
#include <hpc_vector.hpp>
#include <hpc_vector3.hpp>
#include <iostream>
#include <lgr_adapt_util.hpp>
#include <lgr_input.hpp>
#include <lgr_mesh_indices.hpp>
#include <lgr_state.hpp>
#include <otm_adapt.hpp>
#include <otm_adapt_util.hpp>
#include <otm_distance_util.hpp>
#include <otm_meshless.hpp>
#include <otm_search.hpp>

namespace lgr {

void
polar_lie_decompose(
    hpc::device_array_vector<hpc::matrix3x3<double>, point_index> const& F,
    hpc::device_array_vector<hpc::vector3<double>, point_index>&         r,
    hpc::device_array_vector<hpc::matrix3x3<double>, point_index>&       u,
    hpc::counting_range<point_index> const&                              source_range)
{
  auto const points_to_F  = F.cbegin();
  auto const index_to_r   = r.begin();
  auto const index_to_u   = u.begin();
  auto const point_offset = *(source_range.begin());
  auto       polar_lie    = [=] HPC_DEVICE(point_index const point) {
    auto const index           = point - point_offset;
    auto const F               = points_to_F[point].load();
    auto const R               = hpc::polar_rotation(F);
    auto const U               = hpc::symm(hpc::transpose(R) * F);
    auto const rotation_vector = hpc::rotation_vector_from_rotation_tensor(R);
    auto const log_stretch     = hpc::log(U);
    index_to_r[index]          = rotation_vector;
    index_to_u[index]          = log_stretch;
  };
  hpc::for_each(hpc::device_policy(), source_range, polar_lie);
  align_rotation_vectors(r);
}

void
otm_populate_new_nodes(
    state&     s,
    node_index begin_src,
    node_index end_src,
    node_index begin_target,
    node_index end_target)
{
  hpc::counting_range<node_index>                          source_range(begin_src, end_src);
  hpc::counting_range<node_index>                          target_range(begin_target, end_target);
  hpc::device_vector<hpc::basis_value<double>, node_index> NZ(source_range.size());
  auto const                                               nodes_to_x          = s.x.cbegin();
  auto const                                               nodes_to_u          = s.u.begin();
  auto const                                               nodes_to_v          = s.v.begin();
  auto const                                               index_to_NZ         = NZ.begin();
  auto const                                               eps                 = s.maxent_desired_tolerance;
  auto const                                               beta                = s.otm_beta;
  auto                                                     maxent_interpolator = [=] HPC_DEVICE(node_index const node) {
    auto const                  target    = nodes_to_x[node].load();
    auto                        converged = false;
    hpc::basis_gradient<double> mu(0.0, 0.0, 0.0);
    using jacobian = hpc::matrix3x3<hpc::quantity<double, hpc::area_dimension>>;
    auto       J        = jacobian::zero();
    auto       iter     = 0;
    auto const max_iter = 16;
    while (converged == false) {
      HPC_ASSERT(iter < max_iter, "Exceeded maximum iterations");
      hpc::position<double> R(0.0, 0.0, 0.0);
      auto                  dRdmu = jacobian::zero();
      for (auto&& source_node : source_range) {
        auto const r   = nodes_to_x[source_node].load() - target;
        auto const rr  = hpc::inner_product(r, r);
        auto const mur = hpc::inner_product(mu, r);
        auto const boltzmann_factor = std::exp(-mur - beta * rr);
        R += r * boltzmann_factor;
        dRdmu -= boltzmann_factor * hpc::outer_product(r, r);
      }
      auto const dmu = -hpc::solve_full_pivot(dRdmu, R);
      mu += dmu;
      auto const nmu  = hpc::norm(mu);
      auto const ndmu = hpc::norm(dmu);
      auto const error = nmu > hpc::machine_epsilon<double>() ? ndmu / nmu : ndmu;
      converged = (error <= eps);
      J         = dRdmu;
      ++iter;
    }
    auto Z = 0.0;
    auto i = 0;
    for (auto&& source_node : source_range) {
      auto const r   = nodes_to_x[source_node].load() - target;
      auto const rr  = hpc::inner_product(r, r);
      auto const mur = hpc::inner_product(mu, r);
      auto const boltzmann_factor = std::exp(-mur - beta * rr);
      Z += boltzmann_factor;
      index_to_NZ[i] = boltzmann_factor;
      ++i;
    }
    i           = 0;
    auto node_u = hpc::displacement<double>::zero();
    auto node_v = hpc::velocity<double>::zero();
    for (auto&& source_node : source_range) {
      auto const u = nodes_to_u[source_node].load();
      auto const v = nodes_to_v[source_node].load();
      auto const N = index_to_NZ[i] / Z;
      node_u += N * u;
      node_v += N * v;
      ++i;
    }
    nodes_to_u[node] = node_u;
    nodes_to_v[node] = node_v;
  };
  hpc::for_each(hpc::device_policy(), target_range, maxent_interpolator);
}

void
otm_populate_new_points(
    state&      s,
    point_index begin_src,
    point_index end_src,
    point_index begin_target,
    point_index end_target)
{
  hpc::counting_range<point_index>                              source_range(begin_src, end_src);
  hpc::counting_range<point_index>                              target_range(begin_target, end_target);
  auto const                                                    source_size = source_range.size();
  hpc::device_vector<hpc::basis_value<double>, point_index>     NZ(source_size);
  hpc::device_array_vector<hpc::vector3<double>, point_index>   r(source_size);
  hpc::device_array_vector<hpc::matrix3x3<double>, point_index> u(source_size);
  hpc::device_array_vector<hpc::vector3<double>, point_index>   rp(source_size);
  hpc::device_array_vector<hpc::matrix3x3<double>, point_index> up(source_size);
  auto const                                                    points_to_xp     = s.xp.cbegin();
  auto const                                                    points_to_K      = s.K.begin();
  auto const                                                    points_to_G      = s.G.begin();
  auto const                                                    points_to_rho    = s.rho.begin();
  auto const                                                    points_to_ep     = s.ep.begin();
  auto const                                                    points_to_b      = s.b.begin();
  auto const                                                    points_to_V      = s.V.begin();
  auto const                                                    points_to_F      = s.F_total.begin();
  polar_lie_decompose(s.F_total, r, u, source_range);
  auto const points_to_Fp = s.Fp_total.begin();
  polar_lie_decompose(s.Fp_total, rp, up, source_range);
  auto const index_to_NZ         = NZ.begin();
  auto const index_to_r          = r.cbegin();
  auto const index_to_u          = u.cbegin();
  auto const index_to_rp         = rp.cbegin();
  auto const index_to_up         = up.cbegin();
  auto const eps                 = s.maxent_desired_tolerance;
  auto const beta                = s.otm_beta;
  auto       maxent_interpolator = [=] HPC_DEVICE(point_index const point) {
    auto const                  target    = points_to_xp[point].load();
    auto                        converged = false;
    hpc::basis_gradient<double> mu(0.0, 0.0, 0.0);
    using jacobian      = hpc::matrix3x3<hpc::quantity<double, hpc::area_dimension>>;
    auto       J        = jacobian::zero();
    auto       iter     = 0;
    auto const max_iter = 16;
    while (converged == false) {
      HPC_ASSERT(iter < max_iter, "Exceeded maximum iterations");
      hpc::position<double> R(0.0, 0.0, 0.0);
      auto                  dRdmu = jacobian::zero();
      for (auto&& source_point : source_range) {
        auto const r                = points_to_xp[source_point].load() - target;
        auto const rr               = hpc::inner_product(r, r);
        auto const mur              = hpc::inner_product(mu, r);
        auto const boltzmann_factor = std::exp(-mur - beta * rr);
        R += r * boltzmann_factor;
        dRdmu -= boltzmann_factor * hpc::outer_product(r, r);
      }
      auto const dmu = -hpc::solve_full_pivot(dRdmu, R);
      mu += dmu;
      auto const nmu   = hpc::norm(mu);
      auto const ndmu  = hpc::norm(dmu);
      auto const error = nmu > hpc::machine_epsilon<double>() ? ndmu / nmu : ndmu;
      converged        = (error <= eps);
      J                = dRdmu;
      ++iter;
    }
    auto Z = 0.0;
    auto i = 0;
    for (auto&& source_point : source_range) {
      auto const r                = points_to_xp[source_point].load() - target;
      auto const rr               = hpc::inner_product(r, r);
      auto const mur              = hpc::inner_product(mu, r);
      auto const boltzmann_factor = std::exp(-mur - beta * rr);
      Z += boltzmann_factor;
      index_to_NZ[i] = boltzmann_factor;
      ++i;
    }
    i                 = 0;
    auto point_K      = hpc::pressure<double>(0.0);
    auto point_G      = hpc::pressure<double>(0.0);
    auto point_rho    = hpc::density<double>(0.0);
    auto point_ep     = hpc::strain<double>(0.0);
    auto point_b      = hpc::acceleration<double>::zero();
    auto point_V      = hpc::volume<double>(0.0);
    auto index_r      = hpc::vector3<double>::zero();
    auto index_u      = hpc::matrix3x3<double>::zero();
    auto index_rp     = hpc::vector3<double>::zero();
    auto index_up     = hpc::matrix3x3<double>::zero();
    for (auto&& source_point : source_range) {
      auto const K                       = points_to_K[source_point];
      auto const G                       = points_to_G[source_point];
      auto const rho                     = points_to_rho[source_point];
      auto const ep                      = points_to_ep[source_point];
      auto const b                       = points_to_b[source_point].load();
      auto const N                       = index_to_NZ[i] / Z;
      auto const dV                      = points_to_V[source_point] * N / (1.0 + N);
      auto const rotation_vector         = index_to_r[i].load();
      auto const log_stretch             = index_to_u[i].load();
      auto const rotation_vector_plastic = index_to_rp[i].load();
      auto const log_stretch_plastic     = index_to_up[i].load();
      point_K += N * K;
      point_G += N * G;
      point_rho += N * rho;
      point_ep += N * ep;
      point_b += N * b;
      point_V += dV;
      points_to_V[source_point] -= dV;
      index_r += N * rotation_vector;
      index_u += N * log_stretch;
      index_rp += N * rotation_vector_plastic;
      index_up += N * log_stretch_plastic;
      ++i;
    }
    points_to_K[point]          = point_K;
    points_to_G[point]          = point_G;
    points_to_rho[point]        = point_rho;
    points_to_ep[point]         = point_ep;
    points_to_b[point]          = point_b;
    points_to_V[point]          = point_V;
    auto const R                = hpc::rotation_tensor_from_rotation_vector(index_r);
    auto const U                = hpc::exp(index_u);
    auto const def_grad         = R * U;
    points_to_F[point]          = def_grad;
    auto const Rp               = hpc::rotation_tensor_from_rotation_vector(index_rp);
    auto const Up               = hpc::exp(index_up);
    auto const def_grad_plastic = Rp * Up;
    points_to_Fp[point]         = def_grad_plastic;
  };
  hpc::for_each(hpc::device_policy(), target_range, maxent_interpolator);
}

otm_adapt_state::otm_adapt_state(state const& s)
    : node_criteria(s.nodes.size()),
      point_criteria(s.points.size()),
      other_node(s.nodes.size()),
      other_point(s.points.size()),
      node_op(s.nodes.size()),
      point_op(s.points.size()),
      point_counts(s.points.size()),
      node_counts(s.nodes.size()),
      old_points_to_new_points(s.points.size() + point_index(1)),
      old_nodes_to_new_nodes(s.nodes.size() + node_index(1)),
      new_points_to_old_points(),
      new_nodes_to_old_nodes(),
      new_point_nodes_to_nodes(),
      new_points_are_same(),
      new_nodes_are_same(),
      interpolate_from_nodes(),
      interpolate_from_points(),
      new_points(point_index(0)),
      new_nodes(node_index(0))
{
}

namespace {

template <typename Index>
void
resize_and_project_adapt_data(
    const hpc::counting_range<Index>&                           old_range,
    const hpc::device_vector<Index, Index>&                     new_counts,
    hpc::counting_range<Index>&                                 new_range,
    hpc::device_vector<Index, Index>&                           old_to_new,
    hpc::device_vector<bool, Index>&                            new_are_same,
    hpc::device_array_vector<hpc::array<Index, 2, int>, Index>& interpolate_from,
    hpc::device_vector<Index, Index>&                           new_to_old)
{
  auto const num_new = hpc::reduce(hpc::device_policy(), new_counts, Index(0));
  hpc::offset_scan(hpc::device_policy(), new_counts, old_to_new);
  new_range.resize(num_new);
  new_to_old.resize(num_new);
  new_are_same.resize(num_new);
  interpolate_from.resize(num_new);
  project(old_range, old_to_new, new_to_old);
}

}  // anonymous namespace

bool
otm_adapt(const input& in, state& s)
{
  otm_adapt_state a(s);

  evaluate_node_adapt(s, a, in.max_node_neighbor_distance);
  evaluate_point_adapt(s, a, in.max_point_neighbor_distance);
  choose_node_adapt(s, a);
  choose_point_adapt(s, a);
  auto const num_chosen_nodes  = get_num_chosen_for_adapt(a.node_op);
  auto const num_chosen_points = get_num_chosen_for_adapt(a.point_op);

  if (num_chosen_nodes == 0 && num_chosen_points == 0) return false;

  if (in.output_to_command_line) {
    std::cout << "adapting " << num_chosen_nodes << " nodes and " << num_chosen_points << " points" << std::endl;
  }

  resize_and_project_adapt_data(
      s.nodes,
      a.node_counts,
      a.new_nodes,
      a.old_nodes_to_new_nodes,
      a.new_nodes_are_same,
      a.interpolate_from_nodes,
      a.new_nodes_to_old_nodes);
  resize_and_project_adapt_data(
      s.points,
      a.point_counts,
      a.new_points,
      a.old_points_to_new_points,
      a.new_points_are_same,
      a.interpolate_from_points,
      a.new_points_to_old_points);

  apply_node_adapt(s, a);
  apply_point_adapt(s, a);
  interpolate_nodal_data(a, s.x);
  interpolate_nodal_data(a, s.u);
  interpolate_nodal_data(a, s.v);
  interpolate_point_data(a, s.xp);
  interpolate_point_data(a, s.h_otm);
  interpolate_point_data(a, s.K);
  interpolate_point_data(a, s.G);
  interpolate_point_data(a, s.rho);
  if (s.ep.size() > 0) interpolate_point_data(a, s.ep);
  interpolate_point_data(a, s.b);
  lie_interpolate_point_data(a, s.F_total);
  if (s.Fp_total.size() > 0) lie_interpolate_point_data(a, s.Fp_total);
  distribute_point_data(a, s.V);
  s.nodes  = a.new_nodes;
  s.points = a.new_points;
  s.nearest_node_neighbor.resize(s.nodes.size());
  s.nearest_node_neighbor_dist.resize(s.nodes.size());
  s.nearest_point_neighbor.resize(s.points.size());
  s.nearest_point_neighbor_dist.resize(s.points.size());
  search::do_otm_iterative_point_support_search(s, in.minimum_support_size);
  otm_update_neighbor_distances(s);
  otm_allocate_state(in, s);
  otm_set_beta(in.otm_gamma, s);
  otm_update_shape_functions(s);
  for (auto material : in.materials) { otm_update_material_state(in, s, material); }

  return true;
}

}  // namespace lgr
