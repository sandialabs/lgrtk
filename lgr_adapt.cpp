#include <lgr_macros.hpp>
#include <lgr_state.hpp>
#include <lgr_input.hpp>
#include <lgr_adapt.hpp>

namespace lgr {

static void LGR_NOINLINE update_bar_Q(state& s) {
  fill(s.Q, double(1.0));
}

static void LGR_NOINLINE update_triangle_Q(state& s) {
  auto const points_to_V = s.V.cbegin();
  auto const point_nodes_to_grad_N = s.grad_N.cbegin();
  auto const elements_to_Q = s.Q.begin();
  auto const points_to_point_nodes =
    s.points * s.nodes_in_element;
  auto const nodes_in_element = s.nodes_in_element;
  auto const elements_to_points = s.elements * s.points_in_element;
  auto functor = [=] (element_index const element) {
    constexpr point_in_element_index fp(0);
    auto const point = elements_to_points[element][fp];
    auto const point_nodes = points_to_point_nodes[point];
    auto sum_g_i_sq = 0.0;
    for (auto const i : nodes_in_element) {
      vector3<double> const grad_N = point_nodes_to_grad_N[point_nodes[i]];
      auto const g_i_sq = (grad_N * grad_N);
      sum_g_i_sq += g_i_sq;
    }
    auto const A = points_to_V[point];
    elements_to_Q[element] = (A * sum_g_i_sq);
  };
  lgr::for_each(s.elements, functor);
}

static void LGR_NOINLINE update_tetrahedron_Q(state& s) {
  auto const points_to_V = s.V.cbegin();
  auto const point_nodes_to_grad_N = s.grad_N.cbegin();
  auto const elements_to_Q = s.Q.begin();
  auto const points_to_point_nodes =
    s.points * s.nodes_in_element;
  auto const nodes_in_element = s.nodes_in_element;
  auto const elements_to_points = s.elements * s.points_in_element;
  auto functor = [=] (element_index const element) {
    constexpr point_in_element_index fp(0);
    auto const point = elements_to_points[element][fp];
    auto const point_nodes = points_to_point_nodes[point];
    auto sum_g_i_sq = 0.0;
    for (auto const i : nodes_in_element) {
      vector3<double> const grad_N = point_nodes_to_grad_N[point_nodes[i]];
      auto const g_i_sq = (grad_N * grad_N);
      sum_g_i_sq += g_i_sq;
    }
    auto const V = points_to_V[point];
    elements_to_Q[element] = (V * V) * (sum_g_i_sq * sum_g_i_sq * sum_g_i_sq);
  };
  lgr::for_each(s.elements, functor);
}

void update_Q(input const& in, state& s) {
  switch (in.element) {
    case BAR: update_bar_Q(s); break;
    case TRIANGLE: update_triangle_Q(s); break;
    case TETRAHEDRON: update_tetrahedron_Q(s); break;
    case COMPOSITE_TETRAHEDRON: assert(0); break;
  }
}

}
