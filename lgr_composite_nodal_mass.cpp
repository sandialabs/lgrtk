#include <lgr_composite_tetrahedron.hpp>
#include <lgr_composite_inline.hpp>
#include <lgr_state.hpp>

namespace lgr {

namespace composite_tetrahedron {

HPC_HOST_DEVICE inline double get_q(double const x) noexcept {
  return 0.25 * (1.0 - std::sqrt(5.0) + 4.0 * std::sqrt(5.0) * x);
}

HPC_HOST_DEVICE inline vector4<double> get_Q(hpc::vector3<double> const xi) noexcept {
  return vector4<double>(
    get_q(1.0 - xi(0) - xi(1) - xi(2)),
    get_q(xi(0)),
    get_q(xi(1)),
    get_q(xi(2)));
}

HPC_HOST_DEVICE inline hpc::array<hpc::vector3<double>, 12> get_centroids() noexcept {
  hpc::array<hpc::vector3<double>, 12> xi;
  xi[0] = hpc::vector3<double>( 0.125, 0.125, 0.125 );
  xi[1] = hpc::vector3<double>( 0.625, 0.125, 0.125 );
  xi[2] = hpc::vector3<double>( 0.125, 0.625, 0.125 );
  xi[3] = hpc::vector3<double>( 0.125, 0.125, 0.625 );
  xi[4] = hpc::vector3<double>( 0.4375, 0.1875, 0.1875 );
  xi[5] = hpc::vector3<double>( 0.3125, 0.3125, 0.3125 );
  xi[6] = hpc::vector3<double>( 0.1875, 0.1875, 0.4375 );
  xi[7] = hpc::vector3<double>( 0.3125, 0.0625, 0.3125 );
  xi[8] = hpc::vector3<double>( 0.3125, 0.3125, 0.0625 );
  xi[9] = hpc::vector3<double>( 0.1875, 0.4375, 0.1875 );
  xi[10] = hpc::vector3<double>( 0.0625, 0.3125, 0.3125 );
  xi[11] = hpc::vector3<double>( 0.1875, 0.1875, 0.1875 );
  return xi;
}

HPC_HOST_DEVICE hpc::array<hpc::array<double, 10>, 10> get_consistent_mass_matrix(
    hpc::array<hpc::vector3<double>, 10> const node_coords,
    vector4<double> point_densities) noexcept {
  auto const S = get_S();
  O_t O;
  get_O(node_coords, S, O);
  auto const O_det = get_O_det(O);
  auto const C = get_centroids();
  auto const gamma = get_gamma();
  hpc::array<hpc::array<double, 10>, 10> mass;
  for (int i = 0; i < 10; ++i) {
    for (int j = 0; j < 10; ++j) {
      mass[i][j] = 0.0;
    }
  }
  for (int tet = 0; tet < 12; ++tet) {
    auto const c_s = C[tet];
    auto const Q_s = get_Q(c_s);
    auto const rho_s = Q_s * point_densities;
    auto const J_s = O_det[tet];
    auto const gamma_s = gamma[tet];
    for (int i = 0; i < 10; ++i) {
      for (int j = 0; j < 10; ++j) {
        mass[i][j] += (J_s * rho_s) * gamma_s[i][j];
      }
    }
  }
  return mass;
}

HPC_HOST_DEVICE hpc::array<double, 10> lump_mass_matrix(hpc::array<hpc::array<double, 10>, 10> const mass) noexcept {
  hpc::array<double, 10> lumped;
  for (int i = 0; i < 10; ++i) {
    lumped[i] = 0.0;
    for (int j = 0; j < 10; ++j) {
      lumped[i] += mass[i][j];
    }
  }
  return lumped;
}

}

void update_nodal_mass_composite_tetrahedron(state& s, material_index const material) {
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

}
