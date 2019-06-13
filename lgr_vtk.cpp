#include <sstream>
#include <fstream>
#include <iomanip>
#include <iostream>

#include <lgr_vtk.hpp>
#include <lgr_print.hpp>
#include <lgr_input.hpp>
#include <lgr_state.hpp>

namespace lgr {

static void start_vtk_file(std::ostream& stream) {
  stream << "# vtk DataFile Version 3.0\n";
  stream << "vtk output\n";
  stream << "ASCII\n";
  stream << "DATASET UNSTRUCTURED_GRID\n";
}

template <class Quantity, class Index>
static void write_vtk_points(std::ostream& stream,
    hpc::device_array_vector<hpc::vector3<Quantity>, Index> const& x) {
  stream << "POINTS " << int(x.size()) << " double\n";
  for (auto ref : x) {
    stream << hpc::vector3<double>(ref.load()) << "\n";
  }
}

static void write_vtk_cells(std::ostream& stream, input const& in, state const& s) {
  stream << "CELLS " << int(s.elements.size()) << " " << int(s.elements.size() * (s.nodes_in_element.size() + 1)) << "\n";
  auto const elements_to_element_nodes = s.elements * s.nodes_in_element;
  auto const element_nodes_to_nodes = s.elements_to_nodes.cbegin();
  for (auto const element_nodes : elements_to_element_nodes) {
    stream << int(s.nodes_in_element.size());
    for (auto const element_node : element_nodes) {
      node_index const node = element_nodes_to_nodes[element_node];
      stream << " " << int(node);
    }
    stream << "\n";
  }
  stream << "CELL_TYPES " << int(s.elements.size()) << "\n";
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

static void write_vtk_point_data(std::ostream& stream, state const& s) {
  stream << "POINT_DATA " << int(s.nodes.size()) << "\n";
}

template <class Quantity, class Index>
static void write_vtk_scalars(std::ostream& stream, std::string const& name,
    hpc::device_vector<Quantity, Index> const& vec) {
  stream << "SCALARS " << name << " double 1\n";
  stream << "LOOKUP_TABLE default\n";
  for (Quantity const val : vec) {
    stream << double(val) << "\n";
  }
}

static void write_vtk_materials(std::ostream& stream,
    hpc::device_vector<material_index, element_index> const& vec) {
  stream << "SCALARS material int 1\n";
  stream << "LOOKUP_TABLE default\n";
  for (material_index const val : vec) {
    stream << int(val) << "\n";
  }
}

template <class Quantity>
static void write_vtk_scalars(std::ostream& stream, char const* name,
    hpc::counting_range<element_index> const elements,
    hpc::counting_range<point_in_element_index> const points_in_element,
    hpc::device_vector<Quantity, point_index> const& vec) {
  auto const elements_to_points = elements * points_in_element;
  for (auto const qp : points_in_element) {
    std::string suffix = (points_in_element.size() == 1) ? "" : (std::string("_") + std::to_string(int(qp)));
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
    hpc::device_array_vector<hpc::vector3<Quantity>, point_index> const& vec) {
  auto const elements_to_points = elements * points_in_element;
  for (auto const qp : points_in_element) {
    std::string suffix = (int(points_in_element.size()) == 1) ? "" : (std::string("_") + std::to_string(int(qp)));
    stream << "VECTORS " << name << suffix << " double\n";
    for (auto const e : elements) {
      auto const p = elements_to_points[e][qp];
      stream << hpc::vector3<double>(vec.begin()[p].load()) << "\n";
    }
  }
}

template <class Quantity, class Index>
static void write_vtk_vectors(std::ostream& stream, char const* name,
    hpc::device_array_vector<hpc::vector3<Quantity>, Index> const& vec) {
  stream << "VECTORS " << name << " double\n";
  for (auto const ref : vec) {
    stream << hpc::vector3<double>(ref.load()) << "\n";
  }
}

static void write_vtk_cell_data(std::ostream& stream, state const& s) {
  stream << "CELL_DATA " << int(s.elements.size()) << "\n";
}

void file_writer::write(
    input const& in,
    int const file_output_index,
    state const& s
    ) {
  std::stringstream filename_stream;
  filename_stream << prefix << "_" << file_output_index << ".vtk";
  auto const filename = filename_stream.str();
  std::ofstream stream(filename.c_str());
  stream << std::scientific << std::setprecision(17);
  start_vtk_file(stream);
  write_vtk_points(stream, s.x);
  write_vtk_cells(stream, in, s);
  //POINTS
  write_vtk_point_data(stream, s);
  assert(s.x.size() == s.nodes.size());
  write_vtk_vectors(stream, "position", s.x);
  assert(s.v.size() == s.nodes.size());
  write_vtk_vectors(stream, "velocity", s.v);
  for (material_index const material : in.materials) {
    if (in.enable_nodal_pressure[material] || in.enable_nodal_energy[material]) {
      std::stringstream name_stream;
      name_stream << "nodal_pressure_" << int(material);
      auto name = name_stream.str();
      assert(s.p_h[material].size() == s.nodes.size());
      write_vtk_scalars(stream, name, s.p_h[material]);
    }
    if (in.enable_nodal_energy[material]) {
      {
        std::stringstream name_stream;
        name_stream << "nodal_energy_" << int(material);
        auto name = name_stream.str();
        assert(s.e_h[material].size() == s.nodes.size());
        write_vtk_scalars(stream, name, s.e_h[material]);
      }
      {
        std::stringstream name_stream;
        name_stream << "nodal_density_" << int(material);
        auto name = name_stream.str();
        assert(s.rho_h[material].size() == s.nodes.size());
        write_vtk_scalars(stream, name, s.rho_h[material]);
      }
    }
  }
  if (in.enable_adapt) {
    assert(s.h_adapt.size() == s.nodes.size());
    write_vtk_scalars(stream, "h", s.h_adapt);
  }
  //CELLS
  write_vtk_cell_data(stream, s);
  auto have_nodal_pressure_or_energy = [&] (material_index const material) {
    return in.enable_nodal_pressure[material] || in.enable_nodal_energy[material];
  };
  if (!hpc::all_of(hpc::serial_policy(), in.materials, have_nodal_pressure_or_energy)) {
    write_vtk_scalars(stream, "pressure", s.elements, s.points_in_element, s.p);
  }
  if (!hpc::all_of(hpc::serial_policy(), in.enable_nodal_energy)) {
    write_vtk_scalars(stream, "energy", s.elements, s.points_in_element, s.e);
    write_vtk_scalars(stream, "density", s.elements, s.points_in_element, s.rho);
  }
  if (hpc::any_of(hpc::serial_policy(), in.enable_nodal_energy)) {
    write_vtk_vectors(stream, "q", s.elements, s.points_in_element, s.q);
    if (hpc::any_of(hpc::serial_policy(), in.enable_p_prime)) {
      write_vtk_scalars(stream, "p_prime", s.elements, s.points_in_element, s.p_prime);
    }
  }
  write_vtk_scalars(stream, "time_step", s.elements, s.points_in_element, s.element_dt);
  if (in.enable_adapt) {
    write_vtk_scalars(stream, "quality", s.quality);
  }
  write_vtk_materials(stream, s.material);
  stream.close();
}

}
