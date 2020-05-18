#pragma once

#include <hpc_array.hpp>
#include <hpc_array_traits.hpp>
#include <hpc_array_vector.hpp>
#include <hpc_dimensional.hpp>
#include <hpc_macros.hpp>
#include <hpc_matrix3x3.hpp>
#include <hpc_quaternion.hpp>
#include <hpc_range.hpp>
#include <hpc_symmetric3x3.hpp>
#include <hpc_vector.hpp>
#include <hpc_vector3.hpp>
#include <lgr_mesh_indices.hpp>
#include <lgr_state.hpp>
#include <cmath>

namespace lgr {
class state;
}

namespace lgr {

void polar_lie_decompose(
    hpc::device_array_vector<hpc::matrix3x3<double>, point_index> const & F,
    hpc::device_array_vector<hpc::vector3<double>, point_index> & r,
    hpc::device_array_vector<hpc::matrix3x3<double>, point_index> & u,
    hpc::counting_range<point_index> const & source_range);

template <typename T, typename I>
void
align_rotation_vectors(hpc::device_array_vector<T, I> & v)
{
  auto const n = v.size();
  assert(n >= 2);
  hpc::counting_range<I> range(0, n);
  hpc::device_vector<double, I> norms(n);
  auto const alpha = 0.8;
  auto const pi = std::acos(-1.0);
  auto const index_to_v = v.begin();
  auto const index_to_norm = norms.begin();
  auto align_functor = [=] HPC_DEVICE (I const index) {
    auto const v0 = index_to_v[0].load();
    auto const vi = index_to_v[index].load();
    auto const ni = hpc::norm(vi);
    auto const dot = hpc::inner_product(v0, vi);
    if (dot <= -alpha * pi * pi) {
      index_to_v[index] = vi - (2.0 * pi / ni) * vi;
    }
    index_to_norm[index] = ni;
  };
  hpc::for_each(hpc::device_policy(), range, align_functor);
  auto normalize_functor = [=] HPC_DEVICE (I const index) {
    auto const vi = index_to_v[index].load();
    auto const ni = index_to_norm[index];
    index_to_v[index] = vi - (2.0 * pi / ni) * vi;
  };
  auto const sum_norms = hpc::reduce(hpc::device_policy(), norms, 0.0);
  if (sum_norms > n * pi) {
    hpc::for_each(hpc::device_policy(), range, normalize_functor);
  }
}

template <typename NodeIndexArray>
void otm_populate_new_nodes_linear(
    state & s,
    hpc::device_array_vector<NodeIndexArray, node_index> const& interpolate_from_nodes)
{
  auto const source_size = interpolate_from_nodes.size();
  hpc::device_vector<hpc::basis_value<double>, node_index> NZ(source_size);
  auto const nodes_to_source_nodes = interpolate_from_nodes.cbegin();
  auto const nodes_to_x = s.x.cbegin();
  auto const nodes_to_u = s.u.begin();
  auto const nodes_to_v = s.v.begin();
  auto const index_to_NZ = NZ.begin();
  auto const eps = s.maxent_tolerance;
  auto const beta = s.otm_beta;
  auto maxent_interpolator = [=] HPC_DEVICE (node_index const node) {
    auto const source_range = nodes_to_source_nodes[node].load();
    auto const target = nodes_to_x[node].load();
    auto converged = false;
    hpc::basis_gradient<double> mu(0.0, 0.0, 0.0);
    using jacobian = hpc::matrix3x3<hpc::quantity<double, hpc::area_dimension>>;
    auto J = jacobian::zero();
    auto iter = 0;
    auto const max_iter = 16;
    while (converged == false) {
      HPC_ASSERT(iter < max_iter, "Exceeded maximum iterations");
      hpc::position<double> R(0.0, 0.0, 0.0);
      auto dRdmu = jacobian::zero();
      for (auto && source_node : source_range) {
        auto const r = nodes_to_x[source_node].load() - target;
        auto const rr = hpc::inner_product(r, r);
        auto const mur = hpc::inner_product(mu, r);
        auto const boltzmann_factor = std::exp(-mur - beta * rr);
        R += r * boltzmann_factor;
        dRdmu -= boltzmann_factor * hpc::outer_product(r, r);
      }
      auto const dmu = -hpc::solve_full_pivot(dRdmu, R);
      mu += dmu;
      auto const nmu = hpc::norm(mu);
      auto const ndmu = hpc::norm(dmu);
      auto const error = nmu > hpc::machine_epsilon<double>() ? ndmu / nmu : ndmu;
      converged = (error <= eps);
      J = dRdmu;
      ++iter;
    }
    auto Z = 0.0;
    auto i = 0;
    for (auto && source_node : source_range) {
      auto const r = nodes_to_x[source_node].load() - target;
      auto const rr = hpc::inner_product(r, r);
      auto const mur = hpc::inner_product(mu, r);
      auto const boltzmann_factor = std::exp(-mur - beta * rr);
      Z += boltzmann_factor;
      index_to_NZ[i] = boltzmann_factor;
      ++i;
    }
    i = 0;
    auto node_u = hpc::displacement<double>::zero();
    auto node_v = hpc::velocity<double>::zero();
    for (auto && source_node : source_range) {
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
  hpc::for_each(hpc::device_policy(), interpolate_from_nodes, maxent_interpolator);
}

template <typename PointIndexArray>
void otm_populate_new_points_linear(
    state & s,
    hpc::device_array_vector<PointIndexArray, point_index> const& interpolate_from_points)
{
  auto const source_size = interpolate_from_points.size();
  hpc::device_vector<hpc::basis_value<double>, point_index> NZ(source_size);
  hpc::device_array_vector<hpc::vector3<double>, point_index> r(source_size);
  hpc::device_array_vector<hpc::matrix3x3<double>, point_index>  u(source_size);
  hpc::device_array_vector<hpc::vector3<double>, point_index> rp(source_size);
  hpc::device_array_vector<hpc::matrix3x3<double>, point_index>  up(source_size);
  auto const points_to_source_points = interpolate_from_points.cbegin();
  auto const points_to_xp = s.xp.cbegin();
  auto const points_to_K = s.K.begin();
  auto const points_to_G = s.G.begin();
  auto const points_to_rho = s.rho.begin();
  auto const points_to_ep = s.ep.begin();
  auto const points_to_ep_dot = s.ep_dot.begin();
  auto const points_to_b = s.b.begin();
  auto const points_to_V = s.V.begin();
  auto const points_to_F = s.F_total.begin();
  auto const points_to_Fp = s.Fp_total.begin();
  auto const index_to_NZ = NZ.begin();
  auto const index_to_r = r.begin();
  auto const index_to_u = u.cbegin();
  auto const index_to_rp = rp.begin();
  auto const index_to_up = up.cbegin();
  auto const eps = s.maxent_tolerance;
  auto const beta = s.otm_beta;
  auto maxent_interpolator = [=] HPC_DEVICE (point_index const point) {
    auto const source_range = points_to_source_points[point].load();
    auto const target = points_to_xp[point].load();
    auto converged = false;
    hpc::basis_gradient<double> mu(0.0, 0.0, 0.0);
    using jacobian = hpc::matrix3x3<hpc::quantity<double, hpc::area_dimension>>;
    auto J = jacobian::zero();
    auto iter = 0;
    auto const max_iter = 16;
    while (converged == false) {
      HPC_ASSERT(iter < max_iter, "Exceeded maximum iterations");
      hpc::position<double> R(0.0, 0.0, 0.0);
      auto dRdmu = jacobian::zero();
      for (auto && source_point : source_range) {
        auto const r = points_to_xp[source_point].load() - target;
        auto const rr = hpc::inner_product(r, r);
        auto const mur = hpc::inner_product(mu, r);
        auto const boltzmann_factor = std::exp(-mur - beta * rr);
        R += r * boltzmann_factor;
        dRdmu -= boltzmann_factor * hpc::outer_product(r, r);
      }
      auto const dmu = -hpc::solve_full_pivot(dRdmu, R);
      mu += dmu;
      auto const nmu = hpc::norm(mu);
      auto const ndmu = hpc::norm(dmu);
      auto const error = nmu > hpc::machine_epsilon<double>() ? ndmu / nmu : ndmu;
      converged = (error <= eps);
      J = dRdmu;
      ++iter;
    }
    auto Z = 0.0;
    auto i = 0;
    for (auto && source_point : source_range) {
      auto const r = points_to_xp[source_point].load() - target;
      auto const rr = hpc::inner_product(r, r);
      auto const mur = hpc::inner_product(mu, r);
      auto const boltzmann_factor = std::exp(-mur - beta * rr);
      Z += boltzmann_factor;
      index_to_NZ[i] = boltzmann_factor;
      ++i;
    }
    auto const alpha = 0.8;
    auto const pi = std::acos(-1.0);
    auto const sum_norms_rvs = 0.0;
    auto const sum_norms_rvs_plastic = 0.0;
    i = 0;
    for (auto && source_point : source_range) {
      auto const F = points_to_F[source_point].load();
      auto const R = hpc::polar_rotation(F);
      auto const U = hpc::symm(hpc::transpose(R) * F);
      auto const rotation_vector = hpc::rotation_vector_from_rotation_tensor(R);
      auto const log_stretch = hpc::log(U);
      index_to_r[i] = rotation_vector;
      index_to_u[i] = log_stretch;
      auto const nrv = hpc::norm(rotation_vector);
      sum_norms_rvs += nrv;
      auto const rv0 = index_to_r[0].load();
      auto const dot = hpc::inner_product(rv0, rotation_vector);
      if (dot <= -alpha * pi * pi) {
        index_to_r[i] = (1.0 - 2.0 * pi / nrv) * rotation_vector;
      }
      auto const Fp = points_to_Fp[source_point].load();
      auto const Rp = hpc::polar_rotation(Fp);
      auto const Up = hpc::symm(hpc::transpose(Rp) * Fp);
      auto const rotation_vector_plastic = hpc::rotation_vector_from_rotation_tensor(Rp);
      auto const log_stretch_plastic = hpc::log(Up);
      index_to_rp[i] = rotation_vector_plastic;
      index_to_up[i] = log_stretch_plastic;
      auto const nrvp = hpc::norm(rotation_vector_plastic);
      sum_norms_rvs_plastic += nrvp;
      auto const rvp0 = index_to_rp[0].load();
      auto const dotp = hpc::inner_product(rvp0, rotation_vector_plastic);
      if (dotp <= -alpha * pi * pi) {
        index_to_rp[i] = (1.0 - 2.0 * pi / nrvp) * rotation_vector_plastic;
      }
      ++i;
    }
    if (sum_norms_rvs > source_size * pi) {
      for (auto j = 0; j < source_size; ++j) {
        auto const rv = index_to_r[j].load();
        auto const nrv = hpc::norm(rv);
        index_to_r[j] = (1.0 - 2.0 * pi / nrv) * rv;
      }
    }
    if (sum_norms_rvs_plastic > source_size * pi) {
      for (auto j = 0; j < source_size; ++j) {
        auto const rvp = index_to_rp[j].load();
        auto const nrvp = hpc::norm(rvp);
        index_to_rp[j] = (1.0 - 2.0 * pi / nrvp) * rvp;
      }
    }
    auto point_K = hpc::pressure<double>(0.0);
    auto point_G = hpc::pressure<double>(0.0);
    auto point_rho = hpc::density<double>(0.0);
    auto point_ep = hpc::strain<double>(0.0);
    auto point_ep_dot = hpc::strain_rate<double>(0.0);
    auto point_b = hpc::acceleration<double>::zero();
    auto point_V = hpc::volume<double>(0.0);
    auto index_r = hpc::vector3<double>::zero();
    auto index_u = hpc::matrix3x3<double>::zero();
    auto index_rp = hpc::vector3<double>::zero();
    auto index_up = hpc::matrix3x3<double>::zero();
    i = 0;
    for (auto && source_point : source_range) {
      auto const K = points_to_K[source_point];
      auto const G = points_to_G[source_point];
      auto const rho = points_to_rho[source_point];
      auto const ep = points_to_ep[source_point];
      auto const ep_dot = points_to_ep_dot[source_point];
      auto const b = points_to_b[source_point].load();
      auto const N = index_to_NZ[i] / Z;
      auto const dV = points_to_V[source_point] * N / (1.0 + N);
      auto const rotation_vector = index_to_r[i].load();
      auto const log_stretch = index_to_u[i].load();
      auto const rotation_vector_plastic = index_to_rp[i].load();
      auto const log_stretch_plastic = index_to_up[i].load();
      point_K += N * K;
      point_G += N * G;
      point_rho += N * rho;
      point_ep += N * ep;
      point_ep_dot += N * ep_dot;
      point_b += N * b;
      point_V += dV;
      points_to_V[source_point] -= dV;
      index_r += N * rotation_vector;
      index_u += N * log_stretch;
      index_rp += N * rotation_vector_plastic;
      index_up += N * log_stretch_plastic;
      ++i;
    }
    points_to_K[point] = point_K;
    points_to_G[point] = point_G;
    points_to_rho[point] = point_rho;
    points_to_ep[point] = point_ep;
    points_to_ep_dot[point] = point_ep_dot;
    points_to_b[point] = point_b;
    points_to_V[point] = point_V;
    auto const R = hpc::rotation_tensor_from_rotation_vector(index_r);
    auto const U = hpc::exp(index_u);
    auto const def_grad = R * U;
    points_to_F[point] = def_grad;
    auto const Rp = hpc::rotation_tensor_from_rotation_vector(index_rp);
    auto const Up = hpc::exp(index_up);
    auto const def_grad_plastic = Rp * Up;
    points_to_Fp[point] = def_grad_plastic;
  };
  hpc::for_each(hpc::device_policy(), interpolate_from_points, maxent_interpolator);
}



void otm_populate_new_nodes(state & s,
    node_index begin_src, node_index end_src,
    node_index begin_target, node_index end_target);

void otm_populate_new_points(state & s,
    point_index begin_src, point_index end_src,
    point_index begin_target, point_index end_target);

bool otm_adapt(const input& in, state& s);

enum adapt_op {
  NONE,
  SPLIT,
  COLLAPSE,
};

struct otm_adapt_state {
  hpc::device_vector<hpc::length<double>, node_index> node_criteria;
  hpc::device_vector<hpc::length<double>, node_index> point_criteria;
  hpc::device_vector<node_index, node_index> other_node;
  hpc::device_vector<point_index, point_index> other_point;
  hpc::device_vector<adapt_op, node_index> node_op;
  hpc::device_vector<adapt_op, point_index> point_op;
  hpc::device_vector<point_index, point_index> point_counts;
  hpc::device_vector<node_index, node_index> node_counts;
  hpc::device_vector<point_index, point_index> old_points_to_new_points;
  hpc::device_vector<node_index, node_index> old_nodes_to_new_nodes;
  hpc::device_vector<point_index, point_index> new_points_to_old_points;
  hpc::device_vector<node_index, node_index> new_nodes_to_old_nodes;
  hpc::device_vector<node_index, point_node_index> new_point_nodes_to_nodes;
  hpc::device_vector<bool, point_index> new_points_are_same;
  hpc::device_vector<bool, node_index> new_nodes_are_same;
  hpc::device_array_vector<hpc::array<node_index, 2, int>, node_index> interpolate_from_nodes;
  hpc::device_array_vector<hpc::array<point_index, 2, int>, point_index> interpolate_from_points;
  hpc::counting_range<point_index> new_points;
  hpc::counting_range<node_index> new_nodes;

  otm_adapt_state(state const&);
};

} // namespace lgr
