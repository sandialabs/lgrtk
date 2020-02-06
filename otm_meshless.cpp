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

void otm_initialize_V(state& s)
{
  auto const element_nodes_to_nodes = s.elements_to_nodes.cbegin();
  auto const nodes_to_x = s.x.cbegin();
  auto const points_to_V = s.V.begin();
  auto const elements_to_element_nodes = s.elements * s.nodes_in_element;
  auto const elements_to_points = s.elements * s.points_in_element;
  auto functor = [=] HPC_DEVICE (element_index const element) {
    constexpr point_in_element_index fp(0);
    auto const element_nodes = elements_to_element_nodes[element];
    using l_t = node_in_element_index;
    hpc::array<hpc::position<double>, 4> x;
    for (int i = 0; i < 4; ++i) {
      auto const node = element_nodes_to_nodes[element_nodes[l_t(i)]];
      x[i] = nodes_to_x[node].load();
    }
    auto const volume = tetrahedron_volume(x);
    assert(volume > 0.0);
    points_to_V[elements_to_points[element][fp]] = volume;
  };
  hpc::for_each(hpc::device_policy(), s.elements, functor);
}

void otm_initialize_grad_val_N(state& s) {
  hpc::dimensionless<double> gamma(1.5);
  auto const support_nodes_to_nodes = s.point_nodes_to_nodes.cbegin();
  auto const nodes_to_x = s.x.cbegin();
  auto const point_nodes_to_N = s.N.begin();
  auto const point_nodes_to_grad_N = s.grad_N.begin();
  auto const points_to_xm = s.xm.cbegin();
  auto const points_to_h = s.h_otm.cbegin();
  auto const nodes_in_support = s.point_nodes.cbegin();
  auto functor = [=] HPC_DEVICE (point_index const point) {
    auto point_nodes = nodes_in_support[point];
    auto const h = points_to_h[point];
    auto const beta = gamma / h / h;
    auto const xm = points_to_xm[point].load();
#if 0
    std::cout << "point     : " << point << std::endl;
#endif
    // Newton's algorithm
    bool converged = false;
    hpc::basis_gradient<double> mu(0.0, 0.0, 0.0);
    auto const eps = 1024 * hpc::machine_epsilon<double>();
    using jacobian = hpc::matrix3x3<hpc::quantity<double, hpc::area_dimension>>;
    jacobian J = jacobian::zero();
    auto const max_iter = 16;
#if 0
    std::cout << "max_iter   : " << max_iter << std::endl;
#endif
    for (auto iter = 0; iter < max_iter; ++iter) {
      hpc::position<double> R(0.0, 0.0, 0.0);
      jacobian dRdmu = jacobian::zero();
#if 0
      std::cout << "iter               : " << iter << std::endl;
      std::cout << "point_nodes.size() : " << point_nodes.size() << std::endl;
#endif
      for (auto point_node : point_nodes) {
        auto const node = support_nodes_to_nodes[point_node];
        auto const xn = nodes_to_x[node].load();
        auto const r = xn - xm;
        auto const rs = hpc::inner_product(r, r);
        auto const mur = hpc::inner_product(mu, r);
        auto const boltzmann_factor = std::exp(-mur - beta * rs);
#if 0
        std::cout << "point_node   : " << point_node << std::endl;
        std::cout << "node         : " << node << std::endl;
#endif
        R += r * boltzmann_factor;
        dRdmu -= boltzmann_factor * hpc::outer_product(r, r);
      }
      auto const dmu = - hpc::solve_full_pivot(dRdmu, R);
      mu += dmu;
      auto const error = hpc::norm(dmu) / hpc::norm(mu);
      converged = error <= eps;
#if 0
      std::cout << "converged  : " << converged << std::endl;
      std::cout << "iter       : " << iter << std::endl;
      std::cout << "det(J)     : " << hpc::determinant(dRdmu) << std::endl;
      std::cout << "|R|        : " << hpc::norm(R) << std::endl;
      std::cout << "|dmu|      : " << hpc::norm(dmu) << std::endl;
      std::cout << "|mu|       : " << hpc::norm(mu) << std::endl;
      std::cout << "|dmu|/|mu| : " << error << std::endl;
      std::cout << "eps        : " << eps << std::endl;
#endif
      if (converged == true) {
        J = dRdmu;
        break;
      }
    }
    auto Z = 0.0;
    for (auto point_node : point_nodes) {
      auto const node = support_nodes_to_nodes[point_node];
      auto const xn = nodes_to_x[node].load();
      auto const r = xn - xm;
      auto const rs = hpc::inner_product(r, r);
      auto const mur = hpc::inner_product(mu, r);
      auto const boltzmann_factor = std::exp(-mur - beta * rs);
      Z += boltzmann_factor;
      point_nodes_to_N[point_node] = boltzmann_factor;
    }
    for (auto point_node : point_nodes) {
      auto const N = point_nodes_to_N[point_node];
      point_nodes_to_N[point_node] = N / Z;
#if 0
        std::cout << "point_node   : " << point_node << std::endl;
        std::cout << "N            : " << point_nodes_to_N[point_node] << std::endl;
#endif
    }
    for (auto point_node : point_nodes) {
      auto const node = support_nodes_to_nodes[point_node];
      auto const xn = nodes_to_x[node].load();
      auto const r = xn - xm;
      auto const N = point_nodes_to_N[point_node];
      auto const Jinvr = hpc::solve_full_pivot(J, r);
      point_nodes_to_grad_N[point_node] = N * Z * Jinvr;
    }
  };
  hpc::for_each(hpc::device_policy(), s.points, functor);
}

void otm_assemble_internal_force(state& s)
{
  auto const points_to_sigma = s.sigma.cbegin();
  auto const points_to_V = s.V.cbegin();
  auto const point_nodes_to_grad_N = s.grad_N.cbegin();
  auto const supports = s.point_nodes.cbegin();
  auto const nodes_to_f = s.f.begin();
  auto const node_points_to_points = s.node_points_to_points.cbegin();
  auto const nodes_to_node_points = s.node_points.cbegin();
  auto const node_points_to_node_ordinals = s.node_influenced_points_to_supporting_nodes.cbegin();
  auto functor = [=] HPC_DEVICE (node_index const node) {
    auto node_f = hpc::force<double>::zero();
    auto const node_points = nodes_to_node_points[node];
    for (auto const node_point : node_points)
    {
      auto const point = node_points_to_points[node_point];
      auto const sigma = points_to_sigma[point].load();
      auto const V = points_to_V[point];
      auto const point_nodes = supports[point];
      auto node_ordinal = node_points_to_node_ordinals[node_point];
      auto point_node = point_nodes[node_ordinal];
      auto const grad_N = point_nodes_to_grad_N[point_node].load();
      auto const f = -(sigma * grad_N) * V;
      node_f = node_f + f;
    }
    auto node_to_f = nodes_to_f[node].load();
    node_to_f += node_f;
  };
  hpc::for_each(hpc::device_policy(), s.nodes, functor);
}

void otm_assemble_external_force(state& s)
{
  auto const points_to_body_acce = s.b.cbegin();
  auto const points_to_rho = s.rho.cbegin();
  auto const points_to_V = s.V.cbegin();
  auto const point_nodes_to_N = s.N.cbegin();
  auto const supports = s.point_nodes.cbegin();
  auto const nodes_to_f = s.f.begin();
  auto const node_points_to_points = s.node_points_to_points.cbegin();
  auto const nodes_to_node_points = s.node_points.cbegin();
  auto const node_points_to_node_ordinals = s.node_influenced_points_to_supporting_nodes.cbegin();
  auto functor = [=] HPC_DEVICE (node_index const node) {
    auto node_f = hpc::force<double>::zero();
    auto const node_points = nodes_to_node_points[node];
    for (auto const node_point : node_points)
    {
      auto const point = node_points_to_points[node_point];
      auto const body_acce = points_to_body_acce[point].load();
      auto const V = points_to_V[point];
      auto const rho = points_to_rho[point];
      auto const point_nodes = supports[point];
      auto node_ordinal = node_points_to_node_ordinals[node_point];
      auto point_node = point_nodes[node_ordinal];
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
  auto const influences = s.node_points.cbegin();
  auto const node_points_to_points = s.node_points_to_points.cbegin();
  auto const supports = s.point_nodes.cbegin();
  auto const node_points_to_node_ordinals = s.node_influenced_points_to_supporting_nodes.cbegin();
  auto const point_nodes_to_N = s.N.begin();
  auto zero_mass = [=] HPC_DEVICE (node_index const node) {
    node_to_mass[node] = 0.0;
  };
  hpc::for_each(hpc::device_policy(), s.nodes, zero_mass);
  auto functor = [=] HPC_DEVICE (node_index const node) {
#if 0
    std::cout << '\n';
    std::cout << "**** node : " << node << '\n';
#endif
    auto node_m = 0.0;
    auto const node_points = influences[node];
    for (auto const node_point : node_points) {
      auto const point = node_points_to_points[node_point];
      auto const V = points_to_V[point];
      auto const rho = points_to_rho[point];
      auto const point_nodes = supports[point];
      auto node_ordinal = node_points_to_node_ordinals[node_point];
      auto point_node = point_nodes[node_ordinal];
      auto const N = point_nodes_to_N[point_node];
      auto const m = N * rho * V;
      node_m += m;
#if 0
      std::cout << "**** node_point   : " << node_point << '\n';
      std::cout << "**** point        : " << point << '\n';
      std::cout << "**** V            : " << V << '\n';
      std::cout << "**** rho          : " << rho << '\n';
      std::cout << "**** node_ordinal : " << node_ordinal << '\n';
      std::cout << "**** point_node   : " << point_node << '\n';
      std::cout << "**** N            : " << N << '\n';
      std::cout << "**** m            : " << m << '\n';
      std::cout << "**** node_m             : " << node_m << '\n';
      std::cout << '\n';
#endif
    }
    node_to_mass[node] += node_m;
#if 0
    std::cout << "**** node_m             : " << node_m << '\n';
    std::cout << "**** node_to_mass[node] : " << node_to_mass[node] << '\n';
    std::cout << '\n';
#endif
  };
  hpc::for_each(hpc::device_policy(), s.nodes, functor);
}

void otm_update_reference(state&) {
}

}
