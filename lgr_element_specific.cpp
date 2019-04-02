#include <cassert>

#include <lgr_element_specific.hpp>
#include <lgr_macros.hpp>
#include <lgr_state.hpp>
#include <lgr_for_each.hpp>
#include <lgr_input.hpp>
#include <lgr_binary_ops.hpp>
#include <lgr_copy.hpp>
#include <lgr_composite_tetrahedron.hpp>

// REMOVE
#include <iostream>

namespace lgr {

static void LGR_NOINLINE initialize_bar_V(state& s) {
  auto const elems_to_nodes_iterator = s.elements_to_nodes.cbegin();
  auto const nodes_to_x = s.x.cbegin();
  auto const points_to_V = s.V.begin();
  auto const elements_to_element_nodes = s.elements * s.nodes_in_element;
  auto const elements_to_points = s.elements * s.points_in_element;
  auto functor = [=] (element_index const element) {
    constexpr point_in_element_index fp(0);
    auto const element_nodes = elements_to_element_nodes[element];
    using l_t = node_in_element_index;
    auto const node0 = elems_to_nodes_iterator[element_nodes[l_t(0)]];
    auto const node1 = elems_to_nodes_iterator[element_nodes[l_t(1)]];
    vector3<double> const x0 = nodes_to_x[node0];
    vector3<double> const x1 = nodes_to_x[node1];
    double const V = x1(0) - x0(0);
    assert(V > 0.0);
    points_to_V[elements_to_points[element][fp]] = V;
  };
  lgr::for_each(s.elements, functor);
}

static void LGR_NOINLINE initialize_triangle_V(state& s)
{
  auto const element_nodes_to_nodes = s.elements_to_nodes.cbegin();
  auto const nodes_to_x = s.x.cbegin();
  auto const points_to_V = s.V.begin();
  auto const elements_to_element_nodes = s.elements * s.nodes_in_element;
  auto const elements_to_points = s.elements * s.points_in_element;
  auto functor = [=] (element_index const element) {
    constexpr point_in_element_index fp(0);
    auto const element_nodes = elements_to_element_nodes[element];
    using l_t = node_in_element_index;
    auto const node0 = element_nodes_to_nodes[element_nodes[l_t(0)]];
    auto const node1 = element_nodes_to_nodes[element_nodes[l_t(1)]];
    auto const node2 = element_nodes_to_nodes[element_nodes[l_t(2)]];
    vector3<double> const x0 = nodes_to_x[node0];
    vector3<double> const x1 = nodes_to_x[node1];
    vector3<double> const x2 = nodes_to_x[node2];
    double const area = 0.5 * (cross((x1 - x0), (x2 - x0))(2));
    assert(area > 0.0);
    points_to_V[elements_to_points[element][fp]] = area;
  };
  lgr::for_each(s.elements, functor);
}

static void LGR_NOINLINE initialize_tetrahedron_V(state& s)
{
  auto const element_nodes_to_nodes = s.elements_to_nodes.cbegin();
  auto const nodes_to_x = s.x.cbegin();
  auto const points_to_V = s.V.begin();
  auto const elements_to_element_nodes = s.elements * s.nodes_in_element;
  auto const elements_to_points = s.elements * s.points_in_element;
  auto functor = [=] (element_index const element) {
    constexpr point_in_element_index fp(0);
    auto const element_nodes = elements_to_element_nodes[element];
    using l_t = node_in_element_index;
    auto const node0 = element_nodes_to_nodes[element_nodes[l_t(0)]];
    auto const node1 = element_nodes_to_nodes[element_nodes[l_t(1)]];
    auto const node2 = element_nodes_to_nodes[element_nodes[l_t(2)]];
    auto const node3 = element_nodes_to_nodes[element_nodes[l_t(3)]];
    vector3<double> const x0 = nodes_to_x[node0];
    vector3<double> const x1 = nodes_to_x[node1];
    vector3<double> const x2 = nodes_to_x[node2];
    vector3<double> const x3 = nodes_to_x[node3];
    double const volume = (1.0 / 6.0) * (cross((x1 - x0), (x2 - x0)) * (x3 - x0));
    assert(volume > 0.0);
    points_to_V[elements_to_points[element][fp]] = volume;
  };
  lgr::for_each(s.elements, functor);
}

static void LGR_NOINLINE initialize_composite_tetrahedron_V(state& s)
{
  auto const element_nodes_to_nodes = s.elements_to_nodes.cbegin();
  auto const nodes_to_x = s.x.cbegin();
  auto const points_to_V = s.V.begin();
  auto const elements_to_element_nodes = s.elements * s.nodes_in_element;
  auto const elements_to_points = s.elements * s.points_in_element;
  auto const nodes_in_element = s.nodes_in_element;
  auto const points_in_element = s.points_in_element;
  auto functor = [=] (element_index const element) {
    auto const element_nodes = elements_to_element_nodes[element];
    array<vector3<double>, 10> node_coords;
    for (auto const node_in_element : nodes_in_element) {
      auto const node = element_nodes_to_nodes[element_nodes[node_in_element]];
      node_coords[int(node_in_element)] = nodes_to_x[node];
    }
    auto const volumes = composite_tetrahedron::get_volumes(node_coords);
#ifndef NDEBUG
    for (auto const volume : volumes) {
      std::cout << "volume " << volume << '\n';
      assert(volume > 0.0);
    }
#endif
    auto const element_points = elements_to_points[element];
    for (auto const qp : points_in_element) {
      points_to_V[element_points[qp]] = volumes[int(qp)];
    }
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
    case COMPOSITE_TETRAHEDRON: initialize_composite_tetrahedron_V(s); break;
  }
}

static void LGR_NOINLINE initialize_bar_grad_N(state& s) {
  auto const points_to_V = s.V.cbegin();
  auto const point_nodes_to_grad_N = s.grad_N.begin();
  auto const points_to_point_nodes = s.points * s.nodes_in_element;
  auto functor = [=] (point_index const point) {
    double const length = points_to_V[point];
    double const inv_length = 1.0 / length;
    vector3<double> const grad_N0 = vector3<double>(-inv_length, 0.0, 0.0);
    vector3<double> const grad_N1 = vector3<double>(inv_length, 0.0, 0.0);
    auto const point_nodes = points_to_point_nodes[point];
    using l_t = node_in_element_index;
    point_nodes_to_grad_N[point_nodes[l_t(0)]] = grad_N0;
    point_nodes_to_grad_N[point_nodes[l_t(1)]] = grad_N1;
  };
  lgr::for_each(s.points, functor);
}

static void LGR_NOINLINE initialize_triangle_grad_N(state& s) {
  auto const element_nodes_to_nodes = s.elements_to_nodes.cbegin();
  auto const nodes_to_x = s.x.cbegin();
  auto const points_to_V = s.V.cbegin();
  auto const point_nodes_to_grad_N = s.grad_N.begin();
  auto const elements_to_element_nodes = s.elements * s.nodes_in_element;
  auto const points_to_point_nodes = s.points * s.nodes_in_element;
  auto const elements_to_points = s.elements * s.points_in_element;
  auto functor = [=] (element_index const element) {
    constexpr point_in_element_index fp(0);
    auto const element_nodes = elements_to_element_nodes[element];
    auto const point = elements_to_points[element][fp];
    auto const point_nodes = points_to_point_nodes[point];
    using l_t = node_in_element_index;
    auto const node0 = element_nodes_to_nodes[element_nodes[l_t(0)]];
    auto const node1 = element_nodes_to_nodes[element_nodes[l_t(1)]];
    auto const node2 = element_nodes_to_nodes[element_nodes[l_t(2)]];
    vector3<double> node_coords[3];
    node_coords[0] = nodes_to_x[node0];
    node_coords[1] = nodes_to_x[node1];
    node_coords[2] = nodes_to_x[node2];
    vector3<double> edge_vectors[3];
    edge_vectors[0] = node_coords[1] - node_coords[0];
    edge_vectors[1] = node_coords[2] - node_coords[0];
    edge_vectors[2] = node_coords[2] - node_coords[1];
    constexpr vector3<double> z_axis(0.0, 0.0, 1.0);
    double const area = points_to_V[point];
    double const factor = 0.5 * (1.0 / area);
    point_nodes_to_grad_N[point_nodes[l_t(0)]] = cross(z_axis, edge_vectors[2]) * factor;
    point_nodes_to_grad_N[point_nodes[l_t(1)]] = -cross(z_axis, edge_vectors[1]) * factor;
    point_nodes_to_grad_N[point_nodes[l_t(2)]] = cross(z_axis, edge_vectors[0]) * factor;
  };
  lgr::for_each(s.elements, functor);
}

static void LGR_NOINLINE initialize_tetrahedron_grad_N(state& s) {
  auto const element_nodes_to_nodes = s.elements_to_nodes.cbegin();
  auto const nodes_to_x = s.x.cbegin();
  auto const points_to_V = s.V.cbegin();
  auto const point_nodes_to_grad_N = s.grad_N.begin();
  auto const elements_to_element_nodes = s.elements * s.nodes_in_element;
  auto const elements_to_points = s.elements * s.points_in_element;
  auto const points_to_point_nodes = s.points * s.nodes_in_element;
  auto functor = [=] (element_index const element) {
    constexpr point_in_element_index fp(0);
    auto const element_nodes = elements_to_element_nodes[element];
    auto const point = elements_to_points[element][fp];
    auto const point_nodes = points_to_point_nodes[point];
    using l_t = node_in_element_index;
    auto const node0 = element_nodes_to_nodes[element_nodes[l_t(0)]];
    auto const node1 = element_nodes_to_nodes[element_nodes[l_t(1)]];
    auto const node2 = element_nodes_to_nodes[element_nodes[l_t(2)]];
    auto const node3 = element_nodes_to_nodes[element_nodes[l_t(3)]];
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
    double const volume = points_to_V[point];
    double const factor = (1.0 / 6.0) * (1.0 / volume);
    vector3<double> grad_N[4];
    grad_N[0] = cross(ev[4], ev[3]) * factor;
    grad_N[1] = cross(ev[1], ev[2]) * factor;
    grad_N[2] = cross(ev[2], ev[0]) * factor;
    grad_N[3] = cross(ev[0], ev[1]) * factor;
    point_nodes_to_grad_N[point_nodes[l_t(0)]] = grad_N[0];
    point_nodes_to_grad_N[point_nodes[l_t(1)]] = grad_N[1];
    point_nodes_to_grad_N[point_nodes[l_t(2)]] = grad_N[2];
    point_nodes_to_grad_N[point_nodes[l_t(3)]] = grad_N[3];
  };
  lgr::for_each(s.elements, functor);
}

static void LGR_NOINLINE initialize_composite_tetrahedron_grad_N(state& s) {
  auto const element_nodes_to_nodes = s.elements_to_nodes.cbegin();
  auto const nodes_to_x = s.x.cbegin();
  auto const point_nodes_to_grad_N = s.grad_N.begin();
  auto const elements_to_element_nodes = s.elements * s.nodes_in_element;
  auto const elements_to_points = s.elements * s.points_in_element;
  auto const points_to_point_nodes = s.points * s.nodes_in_element;
  auto const nodes_in_element = s.nodes_in_element;
  auto const points_in_element = s.points_in_element;
  auto functor = [=] (element_index const element) {
    auto const element_nodes = elements_to_element_nodes[element];
    array<vector3<double>, 10> node_coords;
    for (auto const node_in_element : nodes_in_element) {
      auto const node = element_nodes_to_nodes[element_nodes[node_in_element]];
      node_coords[int(node_in_element)] = nodes_to_x[node];
    }
    auto const grad_N = composite_tetrahedron::get_basis_gradients(node_coords);
    auto const element_points = elements_to_points[element];
    for (auto const qp : points_in_element) {
      auto const point = element_points[qp];
      auto const point_nodes = points_to_point_nodes[point];
      for (auto const a : nodes_in_element) {
        auto const point_node = point_nodes[a];
        point_nodes_to_grad_N[point_node] = grad_N[int(qp)][int(a)];
      }
    }
  };
  lgr::for_each(s.elements, functor);
}

void initialize_grad_N(
    input const& in,
    state& s) {
  switch (in.element) {
    case BAR: initialize_bar_grad_N(s); break;
    case TRIANGLE: initialize_triangle_grad_N(s); break;
    case TETRAHEDRON: initialize_tetrahedron_grad_N(s); break;
    case COMPOSITE_TETRAHEDRON: initialize_composite_tetrahedron_grad_N(s); break;
  }
}

static void LGR_NOINLINE update_bar_h_min(input const&, state& s) {
  auto const elements_to_points = s.elements * s.points_in_element;
  auto const points_to_V = s.V.cbegin();
  auto const elements_to_h_min = s.h_min.begin();
  auto functor = [=] (element_index const element) {
    constexpr point_in_element_index fp(0);
    auto const point = elements_to_points[element][fp];
    elements_to_h_min[element] = double(points_to_V[point]);
  };
  lgr::for_each(s.elements, functor);
}

static void LGR_NOINLINE update_h_min_height(input const&, state& s) {
  auto const point_nodes_to_grad_N = s.grad_N.cbegin();
  auto const elements_to_h_min = s.h_min.begin();
  auto const points_to_point_nodes =
    s.points * s.nodes_in_element;
  auto const elements_to_points = s.elements * s.points_in_element;
  auto functor = [=] (element_index const element) {
    constexpr point_in_element_index fp(0);
    auto const point = elements_to_points[element][fp];
    double min_height = std::numeric_limits<double>::max();
    auto const point_nodes = points_to_point_nodes[point];
    for (auto const point_node : point_nodes) {
      vector3<double> const grad_N = point_nodes_to_grad_N[point_node];
      auto const height = 1.0 / norm(grad_N);
      min_height = lgr::min(min_height, height);
    }
    elements_to_h_min[element] = min_height;
  };
  lgr::for_each(s.elements, functor);
}

static void LGR_NOINLINE update_triangle_h_min_inball(input const&, state& s) {
  auto const point_nodes_to_grad_N = s.grad_N.cbegin();
  auto const elements_to_h_min = s.h_min.begin();
  auto const points_to_point_nodes =
    s.points * s.nodes_in_element;
  auto const nodes_in_element = s.nodes_in_element;
  auto const elements_to_points = s.elements * s.points_in_element;
  auto functor = [=] (element_index const element) {
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
      vector3<double> const grad_N = point_nodes_to_grad_N[point_nodes[i]];
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
  auto const point_nodes_to_grad_N = s.grad_N.cbegin();
  auto const elements_to_h_min = s.h_min.begin();
  auto const points_to_point_nodes =
    s.points * s.nodes_in_element;
  auto const nodes_in_element = s.nodes_in_element;
  auto const elements_to_points = s.elements * s.points_in_element;
  auto functor = [=] (element_index const element) {
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
    double surface_area_over_thrice_volume = 0.0;
    for (auto const i : nodes_in_element) {
      vector3<double> const grad_N = point_nodes_to_grad_N[point_nodes[i]];
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

static void LGR_NOINLINE update_composite_tetrahedron_h_min(state& s) {
  auto const element_nodes_to_nodes = s.elements_to_nodes.cbegin();
  auto const nodes_to_x = s.x.cbegin();
  auto const elements_to_h_min = s.h_min.begin();
  auto const elements_to_element_nodes = s.elements * s.nodes_in_element;
  auto const nodes_in_element = s.nodes_in_element;
  auto functor = [=] (element_index const element) {
    auto const element_nodes = elements_to_element_nodes[element];
    array<vector3<double>, 10> node_coords;
    for (auto const node_in_element : nodes_in_element) {
      auto const node = element_nodes_to_nodes[element_nodes[node_in_element]];
      node_coords[int(node_in_element)] = nodes_to_x[node];
    }
    auto const h_min = composite_tetrahedron::get_length(node_coords);
    elements_to_h_min[element] = h_min;
  };
  lgr::for_each(s.elements, functor);
}

void update_h_min(input const& in, state& s)
{
  switch (in.element) {
    case BAR: update_bar_h_min(in, s); break;
    case TRIANGLE: update_triangle_h_min(in, s); break;
    case TETRAHEDRON: update_tetrahedron_h_min(in, s); break;
    case COMPOSITE_TETRAHEDRON: update_composite_tetrahedron_h_min(s); break;
  }
}

static void LGR_NOINLINE update_bar_h_art(state& s) {
  auto const elements_to_points = s.elements * s.points_in_element;
  auto const points_to_V = s.V.cbegin();
  auto const elements_to_h_art = s.h_art.begin();
  auto functor = [=] (element_index const element) {
    constexpr point_in_element_index fp(0);
    auto const point = elements_to_points[element][fp];
    elements_to_h_art[element] = double(points_to_V[point]);
  };
  lgr::for_each(s.elements, functor);
}

static void LGR_NOINLINE update_triangle_h_art(state& s) {
  double const C_geom = std::sqrt(4.0 / std::sqrt(3.0));
  auto const points_to_V = s.V.cbegin();
  auto const elements_to_h_art = s.h_art.begin();
  auto const elements_to_points = s.elements * s.points_in_element;
  auto functor = [=] (element_index const element) {
    constexpr point_in_element_index fp(0);
    auto const point = elements_to_points[element][fp];
    double const area = points_to_V[point];
    double const h_art = C_geom * std::sqrt(area);
    elements_to_h_art[element] = h_art;
  };
  lgr::for_each(s.elements, functor);
}

static void LGR_NOINLINE update_tetrahedron_h_art(state& s) {
  double const C_geom = std::cbrt(12.0 / std::sqrt(2.0));
  auto const points_to_V = s.V.cbegin();
  auto const elements_to_h_art = s.h_art.begin();
  auto const elements_to_points = s.elements * s.points_in_element;
  auto functor = [=] (element_index const element) {
    constexpr point_in_element_index fp(0);
    auto const point = elements_to_points[element][fp];
    double const volume = points_to_V[point];
    double const h_art = C_geom * std::cbrt(volume);
    elements_to_h_art[element] = h_art;
  };
  lgr::for_each(s.elements, functor);
}

static void LGR_NOINLINE update_composite_tetrahedron_h_art(state& s) {
  copy(s.h_min, s.h_art);
}

void update_h_art(input const& in, state& s) {
  switch (in.element) {
    case BAR: update_bar_h_art(s); break;
    case TRIANGLE: update_triangle_h_art(s); break;
    case TETRAHEDRON: update_tetrahedron_h_art(s); break;
    case COMPOSITE_TETRAHEDRON: update_composite_tetrahedron_h_art(s); break;
  }
}

}
