#include <lgr_element_specific.hpp>
#include <lgr_macros.hpp>
#include <lgr_state.hpp>
#include <lgr_for_each.hpp>
#include <lgr_input.hpp>
#include <lgr_binary_ops.hpp>
#include <lgr_int_range_product.hpp>
#include <lgr_copy.hpp>

namespace lgr {

static void LGR_NOINLINE initialize_bar_V(state& s) {
  auto const elems_to_nodes_iterator = s.elements_to_nodes.cbegin();
  auto const x_iterator = s.x.cbegin();
  auto const V_iterator = s.V.begin();
  auto functor = [=] (int const element) {
    auto const node0 = elems_to_nodes_iterator[element * 2 + 0];
    auto const node1 = elems_to_nodes_iterator[element * 2 + 1];
    vector3<double> const x0 = x_iterator[node0];
    vector3<double> const x1 = x_iterator[node1];
    double const V = x1(0) - x0(0);
    assert(V > 0.0);
    V_iterator[element] = V;
  };
  lgr::for_each(s.elements, functor);
}

static void LGR_NOINLINE initialize_triangle_V(state& s)
{
  auto const element_nodes_to_nodes = s.elements_to_nodes.cbegin();
  auto const nodes_to_x = s.x.cbegin();
  auto const elements_to_V = s.V.begin();
  auto functor = [=] (int const element) {
    auto const node0 = element_nodes_to_nodes[element * 3 + 0];
    auto const node1 = element_nodes_to_nodes[element * 3 + 1];
    auto const node2 = element_nodes_to_nodes[element * 3 + 2];
    vector3<double> const x0 = nodes_to_x[node0];
    vector3<double> const x1 = nodes_to_x[node1];
    vector3<double> const x2 = nodes_to_x[node2];
    double const area = 0.5 * (cross((x1 - x0), (x2 - x0))(2));
    assert(area > 0.0);
    elements_to_V[element] = area;
  };
  lgr::for_each(s.elements, functor);
}

static void LGR_NOINLINE initialize_tetrahedron_V(state& s)
{
  auto const element_nodes_to_nodes = s.elements_to_nodes.cbegin();
  auto const nodes_to_x = s.x.cbegin();
  auto const elements_to_V = s.V.begin();
  auto functor = [=] (int const element) {
    auto const node0 = element_nodes_to_nodes[element * 4 + 0];
    auto const node1 = element_nodes_to_nodes[element * 4 + 1];
    auto const node2 = element_nodes_to_nodes[element * 4 + 2];
    auto const node3 = element_nodes_to_nodes[element * 4 + 3];
    vector3<double> const x0 = nodes_to_x[node0];
    vector3<double> const x1 = nodes_to_x[node1];
    vector3<double> const x2 = nodes_to_x[node2];
    vector3<double> const x3 = nodes_to_x[node3];
    double const volume = (1.0 / 6.0) * (cross((x1 - x0), (x2 - x0)) * (x3 - x0));
    assert(volume > 0.0);
    elements_to_V[element] = volume;
  };
  lgr::for_each(s.elements, functor);
}

void initialize_V(
    input const& in,
    state& s) {
  switch (in.element) {
    case BAR: initialize_bar_V(s); break;
    case TRIANGLE: initialize_triangle_V(s); break;
    case TETRAHEDRON: initialize_tetrahedron_V(s); break;
  }
}

static void LGR_NOINLINE initialize_bar_grad_N(
    int_range const elements,
    host_vector<int> const& /*elements_to_nodes_vector*/,
    decltype(state::x) const& /*x_vector*/,
    host_vector<double> const& V_vector,
    host_vector<vector3<double>>* grad_N_vector) {
  auto const V_iterator = V_vector.cbegin();
  auto const grad_N_iterator = grad_N_vector->begin();
  auto functor = [=] (int const element) {
    double const length = V_iterator[element];
    double const inv_length = 1.0 / length;
    vector3<double> const grad_N0 = vector3<double>(-inv_length, 0.0, 0.0);
    vector3<double> const grad_N1 = vector3<double>(inv_length, 0.0, 0.0);
    grad_N_iterator[element * 2 + 0] = grad_N0;
    grad_N_iterator[element * 2 + 1] = grad_N1;
  };
  lgr::for_each(elements, functor);
}

static void LGR_NOINLINE initialize_triangle_grad_N(
    int_range const elements,
    host_vector<int> const& elements_to_nodes_vector,
    decltype(state::x) const& x_vector,
    host_vector<double> const& V_vector,
    host_vector<vector3<double>>* grad_N_vector) {
  auto const element_nodes_to_nodes = elements_to_nodes_vector.cbegin();
  auto const nodes_to_x = x_vector.cbegin();
  auto const elements_to_V = V_vector.cbegin();
  auto const element_nodes_to_grad_N = grad_N_vector->begin();
  auto functor = [=] (int const element) {
    auto const node0 = element_nodes_to_nodes[element * 3 + 0];
    auto const node1 = element_nodes_to_nodes[element * 3 + 1];
    auto const node2 = element_nodes_to_nodes[element * 3 + 2];
    vector3<double> node_coords[3];
    node_coords[0] = nodes_to_x[node0];
    node_coords[1] = nodes_to_x[node1];
    node_coords[2] = nodes_to_x[node2];
    vector3<double> edge_vectors[3];
    edge_vectors[0] = node_coords[1] - node_coords[0];
    edge_vectors[1] = node_coords[2] - node_coords[0];
    edge_vectors[2] = node_coords[2] - node_coords[1];
    constexpr vector3<double> z_axis(0.0, 0.0, 1.0);
    double const area = elements_to_V[element];
    double const factor = 0.5 * (1.0 / area);
    element_nodes_to_grad_N[element * 3 + 0] = cross(z_axis, edge_vectors[2]) * factor;
    element_nodes_to_grad_N[element * 3 + 1] = -cross(z_axis, edge_vectors[1]) * factor;
    element_nodes_to_grad_N[element * 3 + 2] = cross(z_axis, edge_vectors[0]) * factor;
  };
  lgr::for_each(elements, functor);
}

static void LGR_NOINLINE initialize_tetrahedron_grad_N(
    int_range const elements,
    host_vector<int> const& elements_to_nodes_vector,
    decltype(state::x) const& x_vector,
    host_vector<double> const& V_vector,
    host_vector<vector3<double>>* grad_N_vector) {
  auto const element_nodes_to_nodes = elements_to_nodes_vector.cbegin();
  auto const nodes_to_x = x_vector.cbegin();
  auto const elements_to_V = V_vector.cbegin();
  auto const enodes_to_grad_N = grad_N_vector->begin();
  auto functor = [=] (int const element) {
    auto const node0 = element_nodes_to_nodes[element * 4 + 0];
    auto const node1 = element_nodes_to_nodes[element * 4 + 1];
    auto const node2 = element_nodes_to_nodes[element * 4 + 2];
    auto const node3 = element_nodes_to_nodes[element * 4 + 3];
    vector3<double> node_coords[4];
    node_coords[0] = nodes_to_x[node0];
    node_coords[1] = nodes_to_x[node1];
    node_coords[2] = nodes_to_x[node2];
    node_coords[3] = nodes_to_x[node3];
    vector3<double> ev[5];
    ev[0] = node_coords[1] - node_coords[0];
    ev[1] = node_coords[2] - node_coords[0];
    ev[2] = node_coords[3] - node_coords[0];
    ev[3] = node_coords[2] - node_coords[1];
    ev[4] = node_coords[3] - node_coords[1];
    double const volume = elements_to_V[element];
    double const factor = (1.0 / 6.0) * (1.0 / volume);
    vector3<double> grad_N[4];
    grad_N[0] = cross(ev[4], ev[3]) * factor;
    grad_N[1] = cross(ev[1], ev[2]) * factor;
    grad_N[2] = cross(ev[2], ev[0]) * factor;
    grad_N[3] = cross(ev[0], ev[1]) * factor;
    enodes_to_grad_N[element * 4 + 0] = grad_N[0];
    enodes_to_grad_N[element * 4 + 1] = grad_N[1];
    enodes_to_grad_N[element * 4 + 2] = grad_N[2];
    enodes_to_grad_N[element * 4 + 3] = grad_N[3];
  };
  lgr::for_each(elements, functor);
}

void initialize_grad_N(
    input const& in,
    state& s) {
  switch (in.element) {
    case BAR: initialize_bar_grad_N(s.elements, s.elements_to_nodes, s.x, s.V, &s.grad_N); break;
    case TRIANGLE: initialize_triangle_grad_N(s.elements, s.elements_to_nodes, s.x, s.V, &s.grad_N); break;
    case TETRAHEDRON: initialize_tetrahedron_grad_N(s.elements, s.elements_to_nodes, s.x, s.V, &s.grad_N); break;
  }
}

static void LGR_NOINLINE update_bar_h_min(input const&, state& s) {
  lgr::copy(s.V, s.h_min);
}

static void LGR_NOINLINE update_h_min_height(input const&, state& s) {
  auto const element_nodes_to_grad_N = s.grad_N.cbegin();
  auto const elements_to_h_min = s.h_min.begin();
  auto const elements_to_element_nodes =
    s.elements * s.nodes_in_element;
  auto functor = [=] (int const element) {
    double min_height = std::numeric_limits<double>::max();
    auto const element_nodes = elements_to_element_nodes[element];
    for (auto const element_node : element_nodes) {
      vector3<double> grad_N = element_nodes_to_grad_N[element_node];
      auto const height = 1.0 / norm(grad_N);
      min_height = lgr::min(min_height, height);
    }
    elements_to_h_min[element] = min_height;
  };
  lgr::for_each(s.elements, functor);
}

static void LGR_NOINLINE update_triangle_h_min_inball(input const&, state& s) {
  auto const element_nodes_to_grad_N = s.grad_N.cbegin();
  auto const elements_to_h_min = s.h_min.begin();
  auto functor = [=] (int const element) {
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
    double perimeter_over_twice_area = 0.0;
    for (int i = 0; i < 3; ++i) {
      vector3<double> grad_N = element_nodes_to_grad_N[element * 3 + i];
      auto const edge_length_over_twice_area = norm(grad_N);
      perimeter_over_twice_area += edge_length_over_twice_area;
    }
    double const radius = 1.0 / perimeter_over_twice_area;
    elements_to_h_min[element] = 2.0 * radius;
  };
  lgr::for_each(s.elements, functor);
}

static void LGR_NOINLINE update_triangle_h_min(input const& in, state& s) {
  switch (in.h_min) {
    case MINIMUM_HEIGHT: update_h_min_height(in, s); break;
    case INBALL_DIAMETER: update_triangle_h_min_inball(in, s); break;
  }
}

static void LGR_NOINLINE update_tetrahedron_h_min_inball(input const&, state& s) {
  auto const element_nodes_to_grad_N = s.grad_N.cbegin();
  auto const elements_to_h_min = s.h_min.begin();
  auto functor = [=] (int const element) {
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
    double surface_area_over_thrice_volume = 0.0;
    for (int i = 0; i < 4; ++i) {
      vector3<double> grad_N = element_nodes_to_grad_N[element * 4 + i];
      auto const face_area_over_thrice_volume = norm(grad_N);
      surface_area_over_thrice_volume += face_area_over_thrice_volume;
    }
    double const radius = 1.0 / surface_area_over_thrice_volume;
    elements_to_h_min[element] = 2.0 * radius;
  };
  lgr::for_each(s.elements, functor);
}

static void LGR_NOINLINE update_tetrahedron_h_min(input const& in, state& s) {
  switch (in.h_min) {
    case MINIMUM_HEIGHT: update_h_min_height(in, s); break;
    case INBALL_DIAMETER: update_tetrahedron_h_min_inball(in, s); break;
  }
}

void update_h_min(input const& in, state& s)
{
  switch (in.element) {
    case BAR: update_bar_h_min(in, s); break;
    case TRIANGLE: update_triangle_h_min(in, s); break;
    case TETRAHEDRON: update_tetrahedron_h_min(in, s); break;
  }
}

static void LGR_NOINLINE update_bar_h_art(state& s) {
  lgr::copy(s.V, s.h_art);
}

static void LGR_NOINLINE update_triangle_h_art(state& s) {
  double const C_geom = std::sqrt(4.0 / std::sqrt(3.0));
  auto const elements_to_V = s.V.cbegin();
  auto const elements_to_h_art = s.h_art.begin();
  auto functor = [=] (int const element) {
    double const area = elements_to_V[element];
    double const h_art = C_geom * std::sqrt(area);
    elements_to_h_art[element] = h_art;
  };
  lgr::for_each(s.elements, functor);
}

static void LGR_NOINLINE update_tetrahedron_h_art(state& s) {
  double const C_geom = std::cbrt(12.0 / std::sqrt(2.0));
  auto const elements_to_V = s.V.cbegin();
  auto const elements_to_h_art = s.h_art.begin();
  auto functor = [=] (int const element) {
    double const volume = elements_to_V[element];
    double const h_art = C_geom * std::cbrt(volume);
    elements_to_h_art[element] = h_art;
  };
  lgr::for_each(s.elements, functor);
}

void update_h_art(input const& in, state& s) {
  switch (in.element) {
    case BAR: update_bar_h_art(s); break;
    case TRIANGLE: update_triangle_h_art(s); break;
    case TETRAHEDRON: update_tetrahedron_h_art(s); break;
  }
}

}
