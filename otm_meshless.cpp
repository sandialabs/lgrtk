#include <hpc_algorithm.hpp>
#include <lgr_state.hpp>
#include <hpc_array.hpp>
#include <hpc_execution.hpp>
#include <hpc_vector3.hpp>
#include <lgr_element_specific_inline.hpp>
#include <otm_meshless.hpp>
#include <iostream>

namespace lgr {

void initialize_meshless_V(state& s)
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

void initialize_meshless_grad_val_N(state& s) {
  hpc::dimensionless<double> gamma(1.5);
  auto const support_nodes_to_nodes = s.points_to_supported_nodes.cbegin();
  auto const nodes_to_x = s.x.cbegin();
  auto const point_nodes_to_N = s.N.begin();
  auto const point_nodes_to_grad_N = s.grad_N.begin();
  auto const points_to_xm = s.xm.cbegin();
  auto const points_to_h = s.h_otm.cbegin();
  auto const nodes_in_support = s.nodes_in_support.cbegin();
  auto functor = [=] HPC_DEVICE (point_index const point) {
    auto point_nodes = nodes_in_support[point];
    auto const h = points_to_h[point];
    auto const beta = gamma / h / h;
    auto const xm = points_to_xm[point].load();
    // Newton's algorithm
    bool converged = false;
    hpc::basis_gradient<double> mu(0.0, 0.0, 0.0);
    auto const eps = 1024 * hpc::machine_epsilon<double>();
    using jacobian = hpc::matrix3x3<hpc::quantity<double, hpc::area_dimension>>;
    jacobian J = jacobian::zero();
    auto const max_iter = 16;
    for (auto iter = 0; iter < max_iter; ++iter) {
      hpc::position<double> R(0.0, 0.0, 0.0);
      jacobian dRdmu = jacobian::zero();
      for (auto point_node : point_nodes) {
        auto const node = support_nodes_to_nodes[point_node];
        auto const xn = nodes_to_x[node].load();
        auto const r = xn - xm;
        auto const rs = hpc::inner_product(r, r);
        auto const mur = hpc::inner_product(mu, r);
        auto const boltzmann_factor = std::exp(-mur - beta * rs);
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

void update_meshless_h_min_inball(input const&, state& s) {
  auto const point_nodes_to_grad_N = s.grad_N.cbegin();
  auto const elements_to_h_min = s.h_min.begin();
  auto const points_to_point_nodes =
    s.points * s.nodes_in_element;
  auto const nodes_in_element = s.nodes_in_element;
  auto const elements_to_points = s.elements * s.points_in_element;
  auto functor = [=] HPC_DEVICE (element_index const element) {
    /* find the radius of the inscribed sphere.
       first fun fact: the volume of a tetrahedron equals one third
       times the radius of the inscribed sphere times the surface area
       of the tetrahedron, where the surface area is the sum of its
       face areas.
       second fun fact: the magnitude of the gradient of the basis function
       of a tetrahedron's node is equal to the area of the opposite face
       divided by thrice the tetrahedron volume
       third fun fact: when solving for the radius, volume cancels out
       of the top and bottom of the division.
     */
    constexpr point_in_element_index fp(0);
    auto const point = elements_to_points[element][fp];
    auto const point_nodes = points_to_point_nodes[point];
    decltype(hpc::area<double>() / hpc::volume<double>()) surface_area_over_thrice_volume = 0.0;
    for (auto const i : nodes_in_element) {
      auto const grad_N = point_nodes_to_grad_N[point_nodes[i]].load();
      auto const face_area_over_thrice_volume = norm(grad_N);
      surface_area_over_thrice_volume += face_area_over_thrice_volume;
    }
    auto const radius = 1.0 / surface_area_over_thrice_volume;
    elements_to_h_min[element] = 2.0 * radius;
  };
  hpc::for_each(hpc::device_policy(), s.elements, functor);
}

void update_meshless_h_art(state& s) {
  double const C_geom = std::cbrt(12.0 / std::sqrt(2.0));
  auto const points_to_V = s.V.cbegin();
  auto const elements_to_h_art = s.h_art.begin();
  auto const elements_to_points = s.elements * s.points_in_element;
  auto functor = [=] HPC_DEVICE (element_index const element) {
    hpc::volume<double> volume = 0.0;
    for (auto const point : elements_to_points[element]) {
      volume += points_to_V[point];
    }
    auto const h_art = C_geom * cbrt(volume);
    elements_to_h_art[element] = h_art;
  };
  hpc::for_each(hpc::device_policy(), s.elements, functor);
}

void assemble_meshless_internal_force(state& s)
{
  auto const points_to_sigma = s.sigma.cbegin();
  auto const points_to_V = s.V.cbegin();
  auto const point_nodes_to_grad_N = s.grad_N.cbegin();
  auto const supports = s.nodes_in_support.cbegin();
  auto const nodes_to_f = s.f.begin();
  auto const node_points_to_points = s.nodes_to_influenced_points.cbegin();
  auto const nodes_to_node_points = s.points_in_influence.cbegin();
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
    nodes_to_f[node] = node_f;
  };
  hpc::for_each(hpc::device_policy(), s.nodes, functor);
}

void update_meshless_nodal_force(state&) {
}

void lump_nodal_mass(state& s) {
  auto const node_to_mass = s.mass.begin();
  auto const point_to_rho = s.rho.cbegin();
  auto const point_to_V = s.V.cbegin();
  auto const influences = s.points_in_influence.cbegin();
  auto const nodes_to_node_points = s.nodes_to_influenced_points.cbegin();
  auto const supports = s.nodes_in_support.cbegin();
  auto const points_to_point_nodes = s.points_to_supported_nodes.cbegin();
  auto const point_nodes_to_N = s.N.begin();
  auto node_to_point_node = [=] HPC_DEVICE (point_index const point, node_index const node) {
    auto const support = supports[point];
    for (auto point_node : support) {
      auto const trial_node = points_to_point_nodes[point_node];
      if (trial_node == node) return point_node;
    }
    return -1;
  };
  auto functor = [=] HPC_DEVICE (node_index const node) {
    auto m = 0.0;
    auto const influence = influences[node];
    for (auto node_point: influence) {
      auto const point = nodes_to_node_points[node_point];
      auto const point_node = node_to_point_node(point, node);
      auto const N = point_nodes_to_N[point_node];
      m += N * point_to_rho[point] * point_to_V[point];
    }
    node_to_mass[node] = m;
  };
  hpc::for_each(hpc::device_policy(), s.nodes, functor);
}

}
