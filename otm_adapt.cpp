#include <hpc_array_vector.hpp>
#include <hpc_dimensional.hpp>
#include <hpc_vector.hpp>
#include <hpc_vector3.hpp>
#include <lgr_mesh_indices.hpp>
#include <otm_adapt.hpp>
#include <lgr_state.hpp>

namespace lgr {

void otm_populate_new_nodes(state & s,
    node_index begin_src, node_index end_src,
    node_index begin_target, node_index end_target)
{
  hpc::counting_range<node_index> source_range(begin_src, end_src);
  hpc::counting_range<node_index> target_range(begin_target, end_target);
  hpc::device_vector<hpc::basis_value<double>, node_index> NZ(source_range.size());
  auto const nodes_to_x = s.x.cbegin();
  auto const nodes_to_u = s.u.begin();
  auto const nodes_to_v = s.v.begin();
  auto const index_to_NZ = NZ.begin();
  auto maxent_interpolator = [=] HPC_DEVICE (node_index const node) {
    auto const target = nodes_to_x[node].load();
    auto const gamma = 1.5;
    auto const h = 1.0;
    auto const eps = 8192 * hpc::machine_epsilon<double>();
    auto const beta = gamma / h / h;
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
      for (auto && source_index : source_range) {
        auto const r = nodes_to_x[source_index].load() - target;
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
    for (auto && source_index : source_range) {
      auto const r = nodes_to_x[source_index].load() - target;
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
    for (auto && source_index : source_range) {
      auto const u = nodes_to_u[source_index].load();
      auto const v = nodes_to_v[source_index].load();
      auto const N = index_to_NZ[i] / Z;
      node_u = node_u + N * u;
      node_v = node_v + N * v;
      ++i;
    }
    nodes_to_u[node] = node_u;
    nodes_to_v[node] = node_v;
  };
  hpc::for_each(hpc::device_policy(), target_range, maxent_interpolator);
}

void otm_populate_new_points(state & s,
    point_index begin_src, point_index end_src,
    point_index begin_target, point_index end_target)
{
  hpc::counting_range<point_index> source_range(begin_src, end_src);
  hpc::counting_range<point_index> target_range(begin_target, end_target);
  hpc::device_vector<hpc::basis_value<double>, point_index> NZ(source_range.size());
  auto const points_to_xp = s.xp.cbegin();
  auto const points_to_K = s.K.begin();
  auto const points_to_G = s.G.begin();
  auto const points_to_rho = s.rho.begin();
  auto const points_to_ep = s.ep.begin();
  auto const points_to_ep_dot = s.ep_dot.begin();
  auto const points_to_b = s.b.begin();
  auto const index_to_NZ = NZ.begin();
  auto maxent_interpolator = [=] HPC_DEVICE (point_index const point) {
    auto const target = points_to_xp[point].load();
    auto const gamma = 1.5;
    auto const h = 1.0;
    auto const eps = 8192 * hpc::machine_epsilon<double>();
    auto const beta = gamma / h / h;
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
      for (auto && source_index : source_range) {
        auto const r = points_to_xp[source_index].load() - target;
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
    for (auto && source_index : source_range) {
      auto const r = points_to_xp[source_index].load() - target;
      auto const rr = hpc::inner_product(r, r);
      auto const mur = hpc::inner_product(mu, r);
      auto const boltzmann_factor = std::exp(-mur - beta * rr);
      Z += boltzmann_factor;
      index_to_NZ[i] = boltzmann_factor;
      ++i;
    }
    i = 0;
    auto point_K = hpc::pressure<double>(0.0);
    auto point_G = hpc::pressure<double>(0.0);
    auto point_rho = hpc::density<double>(0.0);
    auto point_ep = hpc::strain<double>(0.0);
    auto point_ep_dot = hpc::strain_rate<double>(0.0);
    auto point_b = hpc::acceleration<double>::zero();
    for (auto && source_index : source_range) {
      auto const K = points_to_K[source_index];
      auto const G = points_to_G[source_index];
      auto const rho = points_to_rho[source_index];
      auto const ep = points_to_ep[source_index];
      auto const ep_dot = points_to_ep_dot[source_index];
      auto const b = points_to_b[source_index].load();
      auto const N = index_to_NZ[i] / Z;
      point_K = point_K + N * K;
      point_G = point_G + N * G;
      point_rho = point_rho + N * rho;
      point_ep = point_ep + N * ep;
      point_ep_dot = point_ep_dot + N * ep_dot;
      point_b = point_b + N * b;
      ++i;
    }
    points_to_K[point] = point_K;
    points_to_G[point] = point_G;
    points_to_rho[point] = point_rho;
    points_to_ep[point] = point_ep;
    points_to_ep_dot[point] = point_ep_dot;
    points_to_b[point] = point_b;
  };
  hpc::for_each(hpc::device_policy(), target_range, maxent_interpolator);
}

} // namespace lgr
