#include <lgr_bar.hpp>
#include <lgr_triangle.hpp>
#include <lgr_tetrahedron.hpp>
#include <lgr_composite_tetrahedron.hpp>
#include <lgr_input.hpp>
#include <lgr_state.hpp>
#include <lgr_element_specific.hpp>

namespace lgr {

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

HPC_NOINLINE inline void update_h_min_height(input const&, state& s) {
  auto const point_nodes_to_grad_N = s.grad_N.cbegin();
  auto const elements_to_h_min = s.h_min.begin();
  auto const points_to_point_nodes =
    s.points * s.nodes_in_element;
  auto const elements_to_points = s.elements * s.points_in_element;
  auto functor = [=] HPC_DEVICE (element_index const element) {
    constexpr point_in_element_index fp(0);
    auto const point = elements_to_points[element][fp];
    hpc::length<double> min_height = hpc::numeric_limits<double>::max();
    auto const point_nodes = points_to_point_nodes[point];
    for (auto const point_node : point_nodes) {
      auto const grad_N = point_nodes_to_grad_N[point_node].load();
      auto const height = 1.0 / norm(grad_N);
      min_height = hpc::min(min_height, height);
    }
    elements_to_h_min[element] = min_height;
  };
  hpc::for_each(hpc::device_policy(), s.elements, functor);
}

HPC_NOINLINE inline void update_triangle_h_min(input const& in, state& s) {
  switch (in.h_min) {
    case MINIMUM_HEIGHT: update_h_min_height(in, s); break;
    case INBALL_DIAMETER: update_triangle_h_min_inball(in, s); break;
  }
}

HPC_NOINLINE inline void update_tetrahedron_h_min(input const& in, state& s) {
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
    case COMPOSITE_TETRAHEDRON: update_composite_tetrahedron_h_min(s); break;
  }
}

void update_h_art(input const& in, state& s) {
  switch (in.element) {
    case BAR: update_bar_h_art(s); break;
    case TRIANGLE: update_triangle_h_art(s); break;
    case TETRAHEDRON: update_tetrahedron_h_art(s); break;
    case COMPOSITE_TETRAHEDRON: update_tetrahedron_h_art(s); break;
  }
}

HPC_NOINLINE inline void update_nodal_mass_uniform(state& s, material_index const material) {
  auto const nodes_to_node_elements = s.nodes_to_node_elements.cbegin();
  auto const node_elements_to_elements = s.node_elements_to_elements.cbegin();
  auto const points_to_rho = s.rho.cbegin();
  auto const points_to_V = s.V.cbegin();
  assert(s.material_mass[material].size() == s.nodes.size());
  auto const nodes_to_m = s.material_mass[material].begin();
  auto const N = 1.0 / double(int(s.nodes_in_element.size()));
  auto const elements_to_points = s.elements * s.points_in_element;
  auto const elements_to_material = s.material.cbegin();
  auto functor = [=] HPC_DEVICE (node_index const node) {
    hpc::mass<double> m(0.0);
    auto const node_elements = nodes_to_node_elements[node];
    for (auto const node_element : node_elements) {
      element_index const element = node_elements_to_elements[node_element];
      material_index const element_material = elements_to_material[element];
      if (element_material != material) continue;
      for (auto const point : elements_to_points[element]) {
        auto const rho = points_to_rho[point];
        auto const V = points_to_V[point];
        m = m + (rho * V) * N;
      }
    }
    nodes_to_m[node] = m;
  };
  hpc::for_each(hpc::device_policy(), s.node_sets[material], functor);
}

void update_nodal_mass(input const& in, state& s) {
  for (auto const material : in.materials) {
    switch (in.element) {
      case BAR:
      case TRIANGLE:
      case TETRAHEDRON:
        update_nodal_mass_uniform(s, material); break;
      case COMPOSITE_TETRAHEDRON:
        update_nodal_mass_composite_tetrahedron(s, material); break;
    }
  }
  hpc::fill(hpc::device_policy(), s.mass, hpc::mass<double>(0.0));
  for (auto const material : in.materials) {
    auto const nodes_to_total = s.mass.begin();
    auto const nodes_to_partial = s.material_mass[material].cbegin();
    auto functor = [=] HPC_DEVICE (node_index const node) {
      auto m_total = nodes_to_total[node];
      auto const m_partial = nodes_to_partial[node];
      m_total = m_total + m_partial;
      nodes_to_total[node] = m_total;
    };
    hpc::for_each(hpc::device_policy(), s.node_sets[material], functor);
  }
}

}
