#include <hpc_algorithm.hpp>
#include <hpc_execution.hpp>
#include <hpc_index.hpp>
#include <hpc_range.hpp>
#include <hpc_vector.hpp>
#include <hpc_vector3.hpp>
#include <lgr_input.hpp>
#include <lgr_state.hpp>
#include <lgr_vtk.hpp>
#include <lgr_vtk_util.hpp>
#include <cassert>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace lgr {

static void write_vtk_cells(std::ostream& stream, input const& in, captured_state const& s) {
  stream << "CELLS " << s.elements.size() << " " << s.elements.size() * (s.nodes_in_element.size() + 1) << "\n";
  auto const elements_to_element_nodes = s.elements * s.nodes_in_element;
  auto const element_nodes_to_nodes = s.element_nodes_to_nodes.cbegin();
  for (auto const element_nodes : elements_to_element_nodes) {
    stream << s.nodes_in_element.size();
    for (auto const element_node : element_nodes) {
      node_index const node = element_nodes_to_nodes[element_node];
      stream << " " << node;
    }
    stream << "\n";
  }
  stream << "CELL_TYPES " << s.elements.size() << "\n";
  int cell_type = -1;
  switch (in.element) {
    case BAR: cell_type = 3; break;
    case TRIANGLE: cell_type = 5; break;
    case TETRAHEDRON: cell_type = 10; break;
    case COMPOSITE_TETRAHEDRON: cell_type = 24; break;
  }
  for (element_index i(0); i < s.elements.size(); ++i) {
    stream << cell_type << "\n";
  }
}

static void write_vtk_materials(std::ostream& stream,
    hpc::pinned_vector<material_index, element_index> const& vec) {
  stream << "SCALARS material int 1\n";
  stream << "LOOKUP_TABLE default\n";
  for (material_index const val : vec) {
    stream << val << "\n";
  }
}

template <class Quantity>
static void write_vtk_scalars(std::ostream& stream, char const* name,
    hpc::counting_range<element_index> const elements,
    hpc::counting_range<point_in_element_index> const points_in_element,
    hpc::pinned_vector<Quantity, point_index> const& vec) {
  auto const elements_to_points = elements * points_in_element;
  for (auto const qp : points_in_element) {
    std::string suffix = (points_in_element.size() == 1) ? "" : (std::string("_") + std::to_string(hpc::weaken(qp)));
    stream << "SCALARS " << name << suffix << " double 1\n";
    stream << "LOOKUP_TABLE default\n";
    for (auto const e : elements) {
      auto const p = elements_to_points[e][qp];
      stream << double(vec.begin()[p]) << "\n";
    }
  }
}

template <class Quantity>
static void write_vtk_vectors(std::ostream& stream, char const* name,
    hpc::counting_range<element_index> const elements,
    hpc::counting_range<point_in_element_index> const points_in_element,
    hpc::pinned_array_vector<hpc::vector3<Quantity>, point_index> const& vec) {
  auto const elements_to_points = elements * points_in_element;
  for (auto const qp : points_in_element) {
    std::string suffix = (points_in_element.size() == 1) ? "" : (std::string("_") + std::to_string(hpc::weaken(qp)));
    stream << "VECTORS " << name << suffix << " double\n";
    for (auto const e : elements) {
      auto const p = elements_to_points[e][qp];
      stream << hpc::vector3<double>(vec.begin()[p].load()) << "\n";
    }
  }
}

static void write_vtk_cell_data(std::ostream& stream, captured_state const& s) {
  stream << "CELL_DATA " << s.elements.size() << "\n";
}

void file_writer::capture(input const& in, state const& s) {
  captured.nodes = s.nodes;
  captured.elements = s.elements;
  captured.nodes_in_element = s.nodes_in_element;
  captured.points_in_element = s.points_in_element;
  captured.element_nodes_to_nodes.resize(s.elements_to_nodes.size());
  hpc::copy(s.elements_to_nodes, captured.element_nodes_to_nodes);
  captured.x.resize(s.x.size());
  hpc::copy(s.x, captured.x);
  captured.v.resize(s.v.size());
  hpc::copy(s.v, captured.v);
  captured.p_h.resize(s.p_h.size());
  captured.e_h.resize(s.e_h.size());
  captured.rho_h.resize(s.rho_h.size());
  for (material_index const material : in.materials) {
    if (in.enable_nodal_pressure[material] || in.enable_nodal_energy[material]) {
      captured.p_h[material].resize(s.p_h[material].size());
      hpc::copy(s.p_h[material], captured.p_h[material]);
    }
    if (in.enable_nodal_energy[material]) {
      {
        captured.e_h[material].resize(s.e_h[material].size());
        hpc::copy(s.e_h[material], captured.e_h[material]);
      }
      {
        captured.rho_h[material].resize(s.rho_h[material].size());
        hpc::copy(s.rho_h[material], captured.rho_h[material]);
      }
    }
  }
  if (in.enable_adapt) {
    captured.h_adapt.resize(s.h_adapt.size());
    hpc::copy(s.h_adapt, captured.h_adapt);
  }
  auto have_nodal_pressure_or_energy = [&] (material_index const material) {
    return in.enable_nodal_pressure[material] || in.enable_nodal_energy[material];
  };
  if (!hpc::all_of(hpc::serial_policy(), in.materials, have_nodal_pressure_or_energy)) {
    captured.p.resize(s.p.size());
    hpc::copy(s.p, captured.p);
  }
  if (!hpc::all_of(hpc::serial_policy(), in.enable_nodal_energy)) {
    captured.e.resize(s.e.size());
    hpc::copy(s.e, captured.e);
    captured.rho.resize(s.rho.size());
    hpc::copy(s.rho, captured.rho);
  }
  if (hpc::any_of(hpc::serial_policy(), in.enable_nodal_energy)) {
    captured.q.resize(s.q.size());
    hpc::copy(s.q, captured.q);
    if (hpc::any_of(hpc::serial_policy(), in.enable_p_prime)) {
      captured.p_prime.resize(s.p_prime.size());
      hpc::copy(s.p_prime, captured.p_prime);
    }
  }
  captured.element_dt.resize(s.element_dt.size());
  hpc::copy(s.element_dt, captured.element_dt);
  if (in.enable_adapt) {
    captured.quality.resize(s.quality.size());
    hpc::copy(s.quality, captured.quality);
  }
  captured.material.resize(s.material.size());
  hpc::copy(s.material, captured.material);
}

void file_writer::write(
    input const& in,
    int const file_output_index
    ) {
  auto stream = make_vtk_output_stream(prefix, file_output_index);

  start_vtk_unstructured_grid_file(stream);
  write_vtk_points(stream, captured.x);
  write_vtk_cells(stream, in, captured);
  //POINTS
  write_vtk_point_data(stream, captured.nodes);
  assert(captured.x.size() == captured.nodes.size());
  write_vtk_vectors(stream, "position", captured.x);
  assert(captured.v.size() == captured.nodes.size());
  write_vtk_vectors(stream, "velocity", captured.v);
  for (material_index const material : in.materials) {
    if (in.enable_nodal_pressure[material] || in.enable_nodal_energy[material]) {
      std::stringstream name_stream;
      name_stream << "nodal_pressure_" << material;
      auto name = name_stream.str();
      assert(captured.p_h[material].size() == captured.nodes.size());
      write_vtk_scalars(stream, name, captured.p_h[material]);
    }
    if (in.enable_nodal_energy[material]) {
      {
        std::stringstream name_stream;
        name_stream << "nodal_energy_" << material;
        auto name = name_stream.str();
        assert(captured.e_h[material].size() == captured.nodes.size());
        write_vtk_scalars(stream, name, captured.e_h[material]);
      }
      {
        std::stringstream name_stream;
        name_stream << "nodal_density_" << material;
        auto name = name_stream.str();
        assert(captured.rho_h[material].size() == captured.nodes.size());
        write_vtk_scalars(stream, name, captured.rho_h[material]);
      }
    }
  }
  if (in.enable_adapt) {
    assert(captured.h_adapt.size() == captured.nodes.size());
    write_vtk_scalars(stream, "h", captured.h_adapt);
  }
  //CELLS
  write_vtk_cell_data(stream, captured);
  auto have_nodal_pressure_or_energy = [&] (material_index const material) {
    return in.enable_nodal_pressure[material] || in.enable_nodal_energy[material];
  };
  if (!hpc::all_of(hpc::serial_policy(), in.materials, have_nodal_pressure_or_energy)) {
    write_vtk_scalars(stream, "pressure", captured.elements, captured.points_in_element, captured.p);
  }
  if (!hpc::all_of(hpc::serial_policy(), in.enable_nodal_energy)) {
    write_vtk_scalars(stream, "energy", captured.elements, captured.points_in_element, captured.e);
    write_vtk_scalars(stream, "density", captured.elements, captured.points_in_element, captured.rho);
  }
  if (hpc::any_of(hpc::serial_policy(), in.enable_nodal_energy)) {
    write_vtk_vectors(stream, "q", captured.elements, captured.points_in_element, captured.q);
    if (hpc::any_of(hpc::serial_policy(), in.enable_p_prime)) {
      write_vtk_scalars(stream, "p_prime", captured.elements, captured.points_in_element, captured.p_prime);
    }
  }
  write_vtk_scalars(stream, "time_step", captured.elements, captured.points_in_element, captured.element_dt);
  if (in.enable_adapt) {
    write_vtk_scalars(stream, "quality", captured.quality);
  }
  write_vtk_materials(stream, captured.material);
  stream.close();
}

}
