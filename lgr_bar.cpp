#include <lgr_bar.hpp>
#include <lgr_state.hpp>

namespace lgr {

void initialize_bar_V(state& s) {
  auto const elems_to_nodes_iterator = s.elements_to_nodes.cbegin();
  auto const nodes_to_x = s.x.cbegin();
  auto const points_to_V = s.V.begin();
  auto const elements_to_element_nodes = s.elements * s.nodes_in_element;
  auto const elements_to_points = s.elements * s.points_in_element;
  auto functor = [=] HPC_DEVICE (element_index const element) {
    constexpr point_in_element_index fp(0);
    auto const element_nodes = elements_to_element_nodes[element];
    using l_t = node_in_element_index;
    auto const node0 = elems_to_nodes_iterator[element_nodes[l_t(0)]];
    auto const node1 = elems_to_nodes_iterator[element_nodes[l_t(1)]];
    auto const x0 = nodes_to_x[node0].load();
    auto const x1 = nodes_to_x[node1].load();
    auto const V = (x1(0) - x0(0)) * hpc::area<double>(1.0);
    assert(V > 0.0);
    points_to_V[elements_to_points[element][fp]] = V;
  };
  hpc::for_each(hpc::device_policy(), s.elements, functor);
}

void initialize_bar_grad_N(state& s) {
  auto const points_to_V = s.V.cbegin();
  auto const point_nodes_to_grad_N = s.grad_N.begin();
  auto const points_to_point_nodes = s.points * s.nodes_in_element;
  auto functor = [=] HPC_DEVICE (point_index const point) {
    auto const volume = points_to_V[point];
    auto const length = volume / hpc::area<double>(1.0);
    auto const inv_length = 1.0 / length;
    auto const grad_N0 = hpc::basis_gradient<double>(-inv_length, 0.0, 0.0);
    auto const grad_N1 = hpc::basis_gradient<double>(inv_length, 0.0, 0.0);
    auto const point_nodes = points_to_point_nodes[point];
    using l_t = node_in_element_index;
    point_nodes_to_grad_N[point_nodes[l_t(0)]] = grad_N0;
    point_nodes_to_grad_N[point_nodes[l_t(1)]] = grad_N1;
  };
  hpc::for_each(hpc::device_policy(), s.points, functor);
}

void update_bar_h_min(input const&, state& s) {
  auto const elements_to_points = s.elements * s.points_in_element;
  auto const points_to_V = s.V.cbegin();
  auto const elements_to_h_min = s.h_min.begin();
  auto functor = [=] HPC_DEVICE (element_index const element) {
    constexpr point_in_element_index fp(0);
    auto const point = elements_to_points[element][fp];
    elements_to_h_min[element] = double(points_to_V[point]);
  };
  hpc::for_each(hpc::device_policy(), s.elements, functor);
}

void update_bar_h_art(state& s) {
  auto const elements_to_points = s.elements * s.points_in_element;
  auto const points_to_V = s.V.cbegin();
  auto const elements_to_h_art = s.h_art.begin();
  auto functor = [=] HPC_DEVICE (element_index const element) {
    constexpr point_in_element_index fp(0);
    auto const point = elements_to_points[element][fp];
    elements_to_h_art[element] = double(points_to_V[point]);
  };
  hpc::for_each(hpc::device_policy(), s.elements, functor);
}

}
