#include <lgr_state.hpp>
#include <hpc_array.hpp>
#include <lgr_element_specific_inline.hpp>
#include <otm_meshless.hpp>

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

void initialize_meshless_N(state& s) {
  hpc::dimensionless<double> gamma(1.5);
  auto const support_nodes_to_nodes = s.supports_to_nodes.cbegin();
  auto const nodes_to_x = s.x.cbegin();
  auto const point_nodes_to_N = s.N.begin();
  auto const point_nodes_to_grad_N = s.grad_N.begin();
  auto const points_to_xm = s.xm.cbegin();
  auto const points_to_h = s.h_otm.cbegin();
  auto const supports = s.points * s.nodes_in_support;
  auto const num_nodes_in_support = s.nodes_in_support.size();
  auto functor = [=] HPC_DEVICE (point_index const point) {
    auto const support = supports[point];
    auto const h = points_to_h[point];
    auto const beta = gamma / h / h;
    auto const xm = points_to_xm[point].load();
    using NSI = node_in_support_index;
    // Newton's algorithm
    bool converged = false;
    hpc::basis_gradient<double> mu(0.0, 0.0, 0.0);
    auto const eps = hpc::machine_epsilon<double>();
    using jacobian = hpc::matrix3x3<hpc::quantity<double, hpc::area_dimension>>;
    jacobian J = jacobian::zero();
    while (converged == false) {
      hpc::position<double> R(0.0, 0.0, 0.0);
      jacobian dRdmu = jacobian::zero();
      for (auto i = 0; i < num_nodes_in_support; ++i) {
        auto const node = support_nodes_to_nodes[support[NSI(i)]];
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
      converged = hpc::norm(dmu) / hpc::norm(mu) <= eps;
      if (converged == true) J = dRdmu;
    }
    auto Z = 0.0;
    for (auto i = 0; i < num_nodes_in_support; ++i) {
      auto const node = support_nodes_to_nodes[support[NSI(i)]];
      auto const xn = nodes_to_x[node].load();
      auto const r = xn - xm;
      auto const rs = hpc::inner_product(r, r);
      auto const mur = hpc::inner_product(mu, r);
      auto const boltzmann_factor = std::exp(-mur - beta * rs);
      Z += boltzmann_factor;
      point_nodes_to_N[node] = boltzmann_factor;
    }
    for (auto i = 0; i < num_nodes_in_support; ++i) {
      auto const node = support_nodes_to_nodes[support[NSI(i)]];
      auto const N = point_nodes_to_N[node].load();
      point_nodes_to_N[node] = N / Z;
    }
    for (auto i = 0; i < num_nodes_in_support; ++i) {
      auto const node = support_nodes_to_nodes[support[NSI(i)]];
      auto const xn = nodes_to_x[node].load();
      auto const r = xn - xm;
      auto const N = point_nodes_to_N[node].load();
      auto const Jinvr = hpc::solve_full_pivot(J, r);
      point_nodes_to_grad_N[node] = N * Z * Jinvr;
    }
  };
  hpc::for_each(hpc::device_policy(), s.points, functor);
}

void initialize_meshless_grad_N(state& s) {
  auto const element_nodes_to_nodes = s.elements_to_nodes.cbegin();
  auto const nodes_to_x = s.x.cbegin();
  auto const points_to_V = s.V.cbegin();
  auto const point_nodes_to_grad_N = s.grad_N.begin();
  auto const elements_to_element_nodes = s.elements * s.nodes_in_element;
  auto const elements_to_points = s.elements * s.points_in_element;
  auto const points_to_point_nodes = s.points * s.nodes_in_element;
  auto functor = [=] HPC_DEVICE (element_index const element) {
    constexpr point_in_element_index fp(0);
    auto const element_nodes = elements_to_element_nodes[element];
    auto const point = elements_to_points[element][fp];
    auto const point_nodes = points_to_point_nodes[point];
    using l_t = node_in_element_index;
    hpc::array<hpc::position<double>, 4> x;
    for (int i = 0; i < 4; ++i) {
      auto const node = element_nodes_to_nodes[element_nodes[l_t(i)]];
      x[i] = nodes_to_x[node].load();
    }
    auto const volume = points_to_V[point];
    auto const grad_N = tetrahedron_basis_gradients(x, volume);
    for (int i = 0; i < 4; ++i) {
      point_nodes_to_grad_N[point_nodes[l_t(i)]] = grad_N[i];
    }
  };
  hpc::for_each(hpc::device_policy(), s.elements, functor);
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

}
