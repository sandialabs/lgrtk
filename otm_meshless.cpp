#include <cassert>
#include <cmath>
#include <iostream>
#include <hpc_algorithm.hpp>
#include <lgr_state.hpp>
#include <hpc_array.hpp>
#include <hpc_execution.hpp>
#include <hpc_vector3.hpp>
#include <lgr_element_specific_inline.hpp>
#include <otm_meshless.hpp>

namespace lgr {

void otm_initialize_u(state& s)
{
  auto const nodes_to_u = s.u.cbegin();
  auto functor = [=] HPC_DEVICE (node_index const node) {
    auto u = nodes_to_u[node].load();
    u(0) = 0.0;
    u(1) = 1.0;
    u(2) = 2.0;
  };
  hpc::for_each(hpc::device_policy(), s.nodes, functor);
}

void otm_initialize_F(state& s)
{
  auto const points_to_F = s.F_total.cbegin();
  auto functor = [=] HPC_DEVICE (point_index const point) {
    auto F = points_to_F[point].load();
    F = hpc::deformation_gradient<double>::identity();
  };
  hpc::for_each(hpc::device_policy(), s.points, functor);
}

void otm_initialize_grad_val_N(state& s) {
  hpc::dimensionless<double> gamma(1.5);
  auto const point_nodes_to_nodes = s.point_nodes_to_nodes.cbegin();
  auto const nodes_to_x = s.x.cbegin();
  auto const point_nodes_to_N = s.N.begin();
  auto const point_nodes_to_grad_N = s.grad_N.begin();
  auto const points_to_xp = s.xp.cbegin();
  auto const points_to_h = s.h_otm.cbegin();
  auto const points_to_point_nodes = s.points_to_point_nodes.cbegin();
  auto functor = [=] HPC_DEVICE (point_index const point) {
    auto point_nodes = points_to_point_nodes[point];
    auto const h = points_to_h[point];
    auto const beta = gamma / h / h;
    auto const xp = points_to_xp[point].load();
    // Newton's algorithm
    auto converged = false;
    hpc::basis_gradient<double> mu(0.0, 0.0, 0.0);
    auto const eps = 1024 * hpc::machine_epsilon<double>();
    using jacobian = hpc::matrix3x3<hpc::quantity<double, hpc::area_dimension>>;
    auto J = jacobian::zero();
    auto const max_iter = 16;
    for (auto iter = 0; iter < max_iter; ++iter) {
      hpc::position<double> R(0.0, 0.0, 0.0);
      auto dRdmu = jacobian::zero();
      for (auto point_node : point_nodes) {
        auto const node = point_nodes_to_nodes[point_node];
        auto const xn = nodes_to_x[node].load();
        auto const r = xn - xp;
        auto const rr = hpc::inner_product(r, r);
        auto const mur = hpc::inner_product(mu, r);
        auto const boltzmann_factor = std::exp(-mur - beta * rr);
        R += r * boltzmann_factor;
        dRdmu -= boltzmann_factor * hpc::outer_product(r, r);
      }
      auto const dmu = -hpc::solve_full_pivot(dRdmu, R);
      mu += dmu;
      auto const error = hpc::norm(dmu) / hpc::norm(mu);
      converged = error <= eps;
      if (converged == true) {
        J = dRdmu;
        break;
      }
    }
    assert(converged == true);
    auto Z = 0.0;
    for (auto point_node : point_nodes) {
      auto const node = point_nodes_to_nodes[point_node];
      auto const xn = nodes_to_x[node].load();
      auto const r = xn - xp;
      auto const rr = hpc::inner_product(r, r);
      auto const mur = hpc::inner_product(mu, r);
      auto const boltzmann_factor = std::exp(-mur - beta * rr);
      Z += boltzmann_factor;
      point_nodes_to_N[point_node] = boltzmann_factor;
    }
    for (auto point_node : point_nodes) {
      auto const node = point_nodes_to_nodes[point_node];
      auto const xn = nodes_to_x[node].load();
      auto const r = xn - xp;
      auto const Jinvr = hpc::solve_full_pivot(J, r);
      auto const NZ = point_nodes_to_N[point_node];
      point_nodes_to_N[point_node] = NZ / Z;
      point_nodes_to_grad_N[point_node] = NZ * Jinvr;
    }
  };
  hpc::for_each(hpc::device_policy(), s.points, functor);
}

inline void otm_assemble_internal_force(state& s)
{
  auto const points_to_sigma = s.sigma.cbegin();
  auto const points_to_V = s.V.cbegin();
  auto const point_nodes_to_grad_N = s.grad_N.cbegin();
  auto const points_to_point_nodes = s.points_to_point_nodes.cbegin();
  auto const nodes_to_f = s.f.begin();
  auto const node_points_to_points = s.node_points_to_points.cbegin();
  auto const nodes_to_node_points = s.nodes_to_node_points.cbegin();
  auto const node_points_to_point_nodes = s.node_points_to_point_nodes.cbegin();
  auto functor = [=] HPC_DEVICE (node_index const node) {
    auto node_f = hpc::force<double>::zero();
    auto const node_points = nodes_to_node_points[node];
    for (auto const node_point : node_points) {
      auto const point = node_points_to_points[node_point];
      auto const sigma = points_to_sigma[point].load();
      auto const V = points_to_V[point];
      auto const point_nodes = points_to_point_nodes[point];
      auto const point_node = point_nodes[node_points_to_point_nodes[node_point]];
      auto const grad_N = point_nodes_to_grad_N[point_node].load();
      auto const f = -(sigma * grad_N) * V;
      node_f = node_f + f;
    }
    auto node_to_f = nodes_to_f[node].load();
    node_to_f += node_f;
  };
  hpc::for_each(hpc::device_policy(), s.nodes, functor);
}

inline void otm_assemble_external_force(state& s)
{
  auto const points_to_body_acce = s.b.cbegin();
  auto const points_to_rho = s.rho.cbegin();
  auto const points_to_V = s.V.cbegin();
  auto const point_nodes_to_N = s.N.cbegin();
  auto const points_to_point_nodes = s.points_to_point_nodes.cbegin();
  auto const nodes_to_f = s.f.begin();
  auto const node_points_to_points = s.node_points_to_points.cbegin();
  auto const nodes_to_node_points = s.nodes_to_node_points.cbegin();
  auto const node_points_to_point_nodes = s.node_points_to_point_nodes.cbegin();
  auto functor = [=] HPC_DEVICE (node_index const node) {
    auto node_f = hpc::force<double>::zero();
    auto const node_points = nodes_to_node_points[node];
    for (auto const node_point : node_points) {
      auto const point = node_points_to_points[node_point];
      auto const body_acce = points_to_body_acce[point].load();
      auto const V = points_to_V[point];
      auto const rho = points_to_rho[point];
      auto const point_nodes = points_to_point_nodes[point];
      auto const point_node = point_nodes[node_points_to_point_nodes[node_point]];
      auto const N = point_nodes_to_N[point_node];
      auto const m = N * rho * V;
      auto const f = m * body_acce;
      node_f = node_f + f;
    }
    auto node_to_f = nodes_to_f[node].load();
    node_to_f += node_f;
  };
  hpc::for_each(hpc::device_policy(), s.nodes, functor);
}

void otm_update_nodal_force(state& s) {
  auto const nodes_to_f = s.f.begin();
  auto node_f = hpc::force<double>::zero();
  auto functor = [=] HPC_DEVICE (node_index const node) {
    auto node_to_f = nodes_to_f[node].load();
    node_to_f = node_f;
  };
  hpc::for_each(hpc::device_policy(), s.nodes, functor);
  otm_assemble_internal_force(s);
  otm_assemble_external_force(s);
}

void otm_lump_nodal_mass(state& s) {
  auto const node_to_mass = s.mass.begin();
  auto const points_to_rho = s.rho.cbegin();
  auto const points_to_V = s.V.cbegin();
  auto const nodes_to_node_points = s.nodes_to_node_points.cbegin();
  auto const node_points_to_points = s.node_points_to_points.cbegin();
  auto const points_to_point_nodes = s.points_to_point_nodes.cbegin();
  auto const node_points_to_point_nodes = s.node_points_to_point_nodes.cbegin();
  auto const point_nodes_to_N = s.N.begin();
  auto zero_mass = [=] HPC_DEVICE (node_index const node) {
    node_to_mass[node] = 0.0;
  };
  hpc::for_each(hpc::device_policy(), s.nodes, zero_mass);
  auto functor = [=] HPC_DEVICE (node_index const node) {
    auto node_m = 0.0;
    auto const node_points = nodes_to_node_points[node];
    for (auto const node_point : node_points) {
      auto const point = node_points_to_points[node_point];
      auto const V = points_to_V[point];
      auto const rho = points_to_rho[point];
      auto const point_nodes = points_to_point_nodes[point];
      auto const point_node = point_nodes[node_points_to_point_nodes[node_point]];
      auto const N = point_nodes_to_N[point_node];
      auto const m = N * rho * V;
      node_m += m;
    }
    node_to_mass[node] += node_m;
  };
  hpc::for_each(hpc::device_policy(), s.nodes, functor);
}

void otm_update_reference(state& s) {
  auto const point_nodes_to_nodes = s.point_nodes_to_nodes.cbegin();
  auto const points_to_point_nodes = s.points_to_point_nodes.cbegin();
  auto const point_nodes_to_grad_N = s.grad_N.begin();
  auto const nodes_to_u = s.u.cbegin();
  auto const points_to_F_total = s.F_total.begin();
  auto const points_to_V = s.V.begin();
  auto const points_to_rho = s.rho.begin();
  auto functor = [=] HPC_DEVICE (point_index const point) {
    auto const point_nodes = points_to_point_nodes[point];
    auto F_incr = hpc::deformation_gradient<double>::identity();
    for (auto point_node : point_nodes) {
      auto const node = point_nodes_to_nodes[point_node];
      auto const u = nodes_to_u[node].load();
      auto const old_grad_N = point_nodes_to_grad_N[point_node].load();
      F_incr = F_incr + outer_product(u, old_grad_N);
    }
    auto const F_inverse_transpose = transpose(inverse(F_incr));
    for (auto const point_node : point_nodes) {
      auto const old_grad_N = point_nodes_to_grad_N[point_node].load();
      auto const new_grad_N = F_inverse_transpose * old_grad_N;
      point_nodes_to_grad_N[point_node] = new_grad_N;
    }
    auto const old_F_total = points_to_F_total[point].load();
    auto const new_F_total = F_incr * old_F_total;
    points_to_F_total[point] = new_F_total;
    auto const J = determinant(F_incr);
    assert(J > 0.0);
    auto const old_V = points_to_V[point];
    auto const new_V = J * old_V;
    assert(new_V > 0.0);
    points_to_V[point] = new_V;
    auto const old_rho = points_to_rho[point];
    auto const new_rho = old_rho / J;
    points_to_rho[point] = new_rho;
  };
  hpc::for_each(hpc::device_policy(), s.points, functor);
}

}
