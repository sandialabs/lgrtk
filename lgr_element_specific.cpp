#include <lgr_bar.hpp>
#include <lgr_triangle.hpp>
#include <lgr_tetrahedron.hpp>

namespace lgr {

HPC_NOINLINE void initialize_composite_tetrahedron_V(state& s)
{
  auto const element_nodes_to_nodes = s.elements_to_nodes.cbegin();
  auto const nodes_to_x = s.x.cbegin();
  auto const points_to_V = s.V.begin();
  auto const elements_to_element_nodes = s.elements * s.nodes_in_element;
  auto const elements_to_points = s.elements * s.points_in_element;
  auto const nodes_in_element = s.nodes_in_element;
  auto const points_in_element = s.points_in_element;
  auto functor = [=] HPC_DEVICE (element_index const element) {
    auto const element_nodes = elements_to_element_nodes[element];
    hpc::array<hpc::vector3<double>, 10> node_coords;
    for (auto const node_in_element : nodes_in_element) {
      auto const node = element_nodes_to_nodes[element_nodes[node_in_element]];
      node_coords[node_in_element.get()] = nodes_to_x[node].load();
    }
    auto const volumes = composite_tetrahedron::get_volumes(node_coords);
#ifndef NDEBUG
    for (auto const volume : volumes) {
      assert(volume > 0.0);
    }
#endif
    auto const element_points = elements_to_points[element];
    for (auto const qp : points_in_element) {
      points_to_V[element_points[qp]] = volumes[qp.get()];
    }
  };
  hpc::for_each(hpc::device_policy(), s.elements, functor);
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

HPC_NOINLINE void initialize_composite_tetrahedron_grad_N(state& s) {
  auto const element_nodes_to_nodes = s.elements_to_nodes.cbegin();
  auto const nodes_to_x = s.x.cbegin();
  auto const point_nodes_to_grad_N = s.grad_N.begin();
  auto const elements_to_element_nodes = s.elements * s.nodes_in_element;
  auto const elements_to_points = s.elements * s.points_in_element;
  auto const points_to_point_nodes = s.points * s.nodes_in_element;
  auto const nodes_in_element = s.nodes_in_element;
  auto const points_in_element = s.points_in_element;
  auto functor = [=] HPC_DEVICE (element_index const element) {
    auto const element_nodes = elements_to_element_nodes[element];
    hpc::array<hpc::vector3<double>, 10> node_coords;
    for (auto const node_in_element : nodes_in_element) {
      auto const node = element_nodes_to_nodes[element_nodes[node_in_element]];
      node_coords[node_in_element.get()] = nodes_to_x[node].load();
    }
    auto const grad_N = composite_tetrahedron::get_basis_gradients(node_coords);
    auto const element_points = elements_to_points[element];
    for (auto const qp : points_in_element) {
      auto const point = element_points[qp];
      auto const point_nodes = points_to_point_nodes[point];
      for (auto const a : nodes_in_element) {
        auto const point_node = point_nodes[a];
        point_nodes_to_grad_N[point_node] = grad_N[qp.get()][a.get()];
      }
    }
  };
  hpc::for_each(hpc::device_policy(), s.elements, functor);
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

HPC_NOINLINE void update_h_min_height(input const&, state& s) {
  auto const point_nodes_to_grad_N = s.grad_N.cbegin();
  auto const elements_to_h_min = s.h_min.begin();
  auto const points_to_point_nodes =
    s.points * s.nodes_in_element;
  auto const elements_to_points = s.elements * s.points_in_element;
  auto functor = [=] HPC_DEVICE (element_index const element) {
    constexpr point_in_element_index fp(0);
    auto const point = elements_to_points[element][fp];
    double min_height = hpc::numeric_limits<double>::max();
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

HPC_NOINLINE void update_triangle_h_min(input const& in, state& s) {
  switch (in.h_min) {
    case MINIMUM_HEIGHT: update_h_min_height(in, s); break;
    case INBALL_DIAMETER: update_triangle_h_min_inball(in, s); break;
  }
}

HPC_NOINLINE void update_tetrahedron_h_min(input const& in, state& s) {
  switch (in.h_min) {
    case MINIMUM_HEIGHT: update_h_min_height(in, s); break;
    case INBALL_DIAMETER: update_tetrahedron_h_min_inball(in, s); break;
  }
}

HPC_NOINLINE void update_composite_tetrahedron_h_min(state& s) {
  auto const element_nodes_to_nodes = s.elements_to_nodes.cbegin();
  auto const nodes_to_x = s.x.cbegin();
  auto const elements_to_h_min = s.h_min.begin();
  auto const elements_to_element_nodes = s.elements * s.nodes_in_element;
  auto const nodes_in_element = s.nodes_in_element;
  auto functor = [=] HPC_DEVICE (element_index const element) {
    auto const element_nodes = elements_to_element_nodes[element];
    hpc::array<hpc::vector3<double>, 10> node_coords;
    for (auto const node_in_element : nodes_in_element) {
      auto const node = element_nodes_to_nodes[element_nodes[node_in_element]];
      node_coords[node_in_element.get()] = nodes_to_x[node].load();
    }
    auto const h_min = composite_tetrahedron::get_length(node_coords);
    elements_to_h_min[element] = h_min;
  };
  hpc::for_each(hpc::device_policy(), s.elements, functor);
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

HPC_NOINLINE void update_nodal_mass_uniform(state& s, material_index const material) {
  auto const nodes_to_node_elements = s.nodes_to_node_elements.cbegin();
  auto const node_elements_to_elements = s.node_elements_to_elements.cbegin();
  auto const points_to_rho = s.rho.cbegin();
  auto const points_to_V = s.V.cbegin();
  assert(s.material_mass[material].size() == s.nodes.size());
  auto const nodes_to_m = s.material_mass[material].begin();
  auto const N = 1.0 / double(s.nodes_in_element.size().get());
  auto const elements_to_points = s.elements * s.points_in_element;
  auto const elements_to_material = s.material.cbegin();
  auto functor = [=] HPC_DEVICE (node_index const node) {
    double m(0.0);
    auto const node_elements = nodes_to_node_elements[node];
    for (auto const node_element : node_elements) {
      element_index const element = node_elements_to_elements[node_element];
      material_index const element_material = elements_to_material[element];
      if (element_material != material) continue;
      for (auto const point : elements_to_points[element]) {
        double const rho = points_to_rho[point];
        double const V = points_to_V[point];
        m = m + (rho * V) * N;
      }
    }
    nodes_to_m[node] = m;
  };
  hpc::for_each(hpc::device_policy(), s.node_sets[material], functor);
}

HPC_NOINLINE void update_nodal_mass_composite_tetrahedron(state& s, material_index const material) {
  auto const nodes_to_node_elements = s.nodes_to_node_elements.cbegin();
  auto const node_elements_to_elements = s.node_elements_to_elements.cbegin();
  auto const points_to_rho = s.rho.cbegin();
  auto const nodes_to_x = s.x.cbegin();
  auto const nodes_to_m = s.material_mass[material].begin();
  auto const elements_to_points = s.elements * s.points_in_element;
  auto const elements_to_element_nodes = s.elements * s.nodes_in_element;
  auto const nodes_in_element = s.nodes_in_element;
  auto const element_nodes_to_nodes = s.elements_to_nodes.cbegin();
  auto const points_in_element = s.points_in_element;
  auto const node_elements_to_nodes_in_element = s.node_elements_to_nodes_in_element.cbegin();
  auto const elements_to_material = s.material.cbegin();
  auto functor = [=] HPC_DEVICE (node_index const node) {
    double m(0.0);
    auto const node_elements = nodes_to_node_elements[node];
    for (auto const node_element : node_elements) {
      element_index const element = node_elements_to_elements[node_element];
      material_index const element_material = elements_to_material[element];
      if (element_material != material) continue;
      node_in_element_index const node_in_element = node_elements_to_nodes_in_element[node_element];
      auto const element_nodes = elements_to_element_nodes[element];
      hpc::array<hpc::vector3<double>, 10> node_coords;
      for (auto const node_in_element2 : nodes_in_element) {
        auto const node2 = element_nodes_to_nodes[element_nodes[node_in_element2]];
        node_coords[node_in_element2.get()] = nodes_to_x[node2].load();
      }
      vector4<double> point_densities;
      auto const element_points = elements_to_points[element];
      for (auto const point_in_element : points_in_element) {
        auto const point = element_points[point_in_element];
        point_densities(point_in_element.get()) = points_to_rho[point];
      }
      auto const coef = composite_tetrahedron::lump_mass_matrix(
          composite_tetrahedron::get_consistent_mass_matrix(node_coords, point_densities));
      m = m + coef[node_in_element.get()];
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
  hpc::fill(hpc::device_policy(), s.mass, double(0.0));
  for (auto const material : in.materials) {
    auto const nodes_to_total = s.mass.begin();
    auto const nodes_to_partial = s.material_mass[material].cbegin();
    auto functor = [=] HPC_DEVICE (node_index const node) {
      double m_total = nodes_to_total[node];
      double const m_partial = nodes_to_partial[node];
      m_total = m_total + m_partial;
      nodes_to_total[node] = m_total;
    };
    hpc::for_each(hpc::device_policy(), s.node_sets[material], functor);
  }
}

}
