#include <lgr_triangle.hpp>
#include <lgr_state.hpp>
#include <hpc_array.hpp>
#include <lgr_element_specific_inline.hpp>

namespace lgr {

void initialize_triangle_V(state& s)
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
    hpc::array<node_index, 3> nodes;
    hpc::array<hpc::vector3<double>, 3> x;
    for (int i = 0; i < 3; ++i) {
      node_index const node = element_nodes_to_nodes[element_nodes[l_t(i)]];
      nodes[i] = node;
      x[i] = nodes_to_x[node].load();
    }
    double const area = triangle_area(x);
    assert(area > 0.0);
    points_to_V[elements_to_points[element][fp]] = area;
  };
  hpc::for_each(hpc::device_policy(), s.elements, functor);
}

void initialize_triangle_grad_N(state& s) {
  auto const element_nodes_to_nodes = s.elements_to_nodes.cbegin();
  auto const nodes_to_x = s.x.cbegin();
  auto const points_to_V = s.V.cbegin();
  auto const point_nodes_to_grad_N = s.grad_N.begin();
  auto const elements_to_element_nodes = s.elements * s.nodes_in_element;
  auto const points_to_point_nodes = s.points * s.nodes_in_element;
  auto const elements_to_points = s.elements * s.points_in_element;
  auto functor = [=] HPC_DEVICE (element_index const element) {
    constexpr point_in_element_index fp(0);
    auto const element_nodes = elements_to_element_nodes[element];
    auto const point = elements_to_points[element][fp];
    auto const point_nodes = points_to_point_nodes[point];
    using l_t = node_in_element_index;
    hpc::array<hpc::vector3<double>, 3> x;
    for (int i = 0; i < 3; ++i) {
      node_index const node = element_nodes_to_nodes[element_nodes[l_t(i)]];
      x[i] = nodes_to_x[node].load();
    }
    double const area = points_to_V[point];
    auto const grad_N = triangle_basis_gradients(x, area);
    for (int i = 0; i < 3; ++i) {
      point_nodes_to_grad_N[point_nodes[l_t(i)]] = grad_N[i];
    }
  };
  hpc::for_each(hpc::device_policy(), s.elements, functor);
}

void update_triangle_h_min_inball(input const&, state& s) {
  auto const point_nodes_to_grad_N = s.grad_N.cbegin();
  auto const elements_to_h_min = s.h_min.begin();
  auto const points_to_point_nodes =
    s.points * s.nodes_in_element;
  auto const nodes_in_element = s.nodes_in_element;
  auto const elements_to_points = s.elements * s.points_in_element;
  auto functor = [=] HPC_DEVICE (element_index const element) {
    /* find the radius of the inscribed circle.
       first fun fact: the area of a triangle equals one half
       times the radius of the inscribed circle times the perimeter
       of the triangle, where the perimeter is the sum of its
       edge lengths.
       second fun fact: the magnitude of the gradient of the basis function
       of a triangle's node is equal to the length of the opposite edge
       divided by twice the triangle area
       third fun fact: when solving for the radius, area cancels out
       of the top and bottom of the division.
     */
    constexpr point_in_element_index fp(0);
    auto const point = elements_to_points[element][fp];
    auto const point_nodes = points_to_point_nodes[point];
    double perimeter_over_twice_area = 0.0;
    for (auto const i : nodes_in_element) {
      auto const grad_N = point_nodes_to_grad_N[point_nodes[i]].load();
      auto const edge_length_over_twice_area = norm(grad_N);
      perimeter_over_twice_area += edge_length_over_twice_area;
    }
    double const radius = 1.0 / perimeter_over_twice_area;
    elements_to_h_min[element] = 2.0 * radius;
  };
  hpc::for_each(hpc::device_policy(), s.elements, functor);
}

void update_triangle_h_art(state& s) {
  double const C_geom = std::sqrt(4.0 / std::sqrt(3.0));
  auto const points_to_V = s.V.cbegin();
  auto const elements_to_h_art = s.h_art.begin();
  auto const elements_to_points = s.elements * s.points_in_element;
  auto functor = [=] HPC_DEVICE (element_index const element) {
    double area = 0.0;
    for (auto const point : elements_to_points[element]) {
      area += points_to_V[point];
    }
    double const h_art = C_geom * std::sqrt(area);
    elements_to_h_art[element] = h_art;
  };
  hpc::for_each(hpc::device_policy(), s.elements, functor);
}

}
