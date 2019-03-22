#include <sstream>
#include <fstream>

#include <lgr_vtk.hpp>
#include <lgr_print.hpp>
#include <lgr_int_range_product.hpp>
#include <lgr_host_vector.hpp>
#include <lgr_input.hpp>
#include <lgr_state.hpp>

namespace lgr {

static void start_vtk_file(std::ostream& stream) {
  print(stream, "# vtk DataFile Version 3.0\n");
  print(stream, "vtk output\n");
  print(stream, "ASCII\n");
  print(stream, "DATASET UNSTRUCTURED_GRID\n");
}

static void write_vtk_points(std::ostream& stream,
    host_vector<vector3<double>> const& x) {
  print(stream, "POINTS ", x.size(), " double\n");
  for (vector3<double> p : x) {
    print(stream, p, "\n");
  }
}

static void write_vtk_cells(std::ostream& stream,
    input const& in,
    int_range const& elements,
    int_range const& nodes_in_element,
    host_vector<int> const& element_nodes_to_nodes_vector) {
  print(stream, "CELLS ", elements.size(), " ", elements.size() * (nodes_in_element.size() + 1), "\n");
  auto const elements_to_element_nodes = elements * nodes_in_element;
  auto const element_nodes_to_nodes = element_nodes_to_nodes_vector.cbegin();
  for (auto const element_nodes : elements_to_element_nodes) {
    print(stream, element_nodes.size());
    for (auto const element_node : element_nodes) {
      int const node = element_nodes_to_nodes[element_node];
      print(stream, " ", node);
    }
    print(stream, "\n");
  }
  print(stream, "CELL_TYPES ", elements.size(), "\n");
  signed char cell_type = -1;
  switch (in.element) {
    case BAR: cell_type = 3; break;
    case TRIANGLE: cell_type = 5; break;
    case TETRAHEDRON: cell_type = 10; break;
  }
  for (int i = 0; i < elements.size(); ++i) {
    print(stream, cell_type, "\n");
  }
}

static void write_vtk_point_data(std::ostream& stream, int_range const& nodes) {
  print(stream, "POINT_DATA ", nodes.size(), "\n");
}

static void write_vtk_scalars(std::ostream& stream, char const* name, host_vector<double> const& vec) {
  print(stream, "SCALARS ", name, " double 1\n");
  print(stream, "LOOKUP_TABLE default\n");
  for (double const val : vec) {
    print(stream, val, "\n");
  }
}

static void write_vtk_vectors(std::ostream& stream, char const* name, host_vector<vector3<double>> const& vec) {
  print(stream, "VECTORS ", name, " double\n");
  for (vector3<double> const val : vec) {
    print(stream, val, "\n");
  }
}

static void write_vtk_cell_data(std::ostream& stream, int_range const& elements) {
  print(stream, "CELL_DATA ", elements.size(), "\n");
}

void file_writer::operator()(
    input const& in,
    int const file_output_index,
    state const& s
    ) {
  std::stringstream filename_stream;
  print(filename_stream, prefix, "_", file_output_index, ".vtk");
  auto const filename = filename_stream.str();
  std::ofstream stream(filename.c_str());
  start_vtk_file(stream);
  write_vtk_points(stream, s.x);
  write_vtk_cells(stream, in, s.elements, s.nodes_in_element, s.elements_to_nodes);
  write_vtk_point_data(stream, s.nodes);
  write_vtk_vectors(stream, "position", s.x);
  write_vtk_vectors(stream, "velocity", s.v);
  if (in.enable_nodal_pressure) {
    write_vtk_scalars(stream, "nodal_pressure", s.p_h);
  }
  write_vtk_cell_data(stream, s.elements);
  write_vtk_scalars(stream, "specific_internal_energy", s.e);
  write_vtk_scalars(stream, "pressure", s.p);
  write_vtk_scalars(stream, "volume", s.V);
  if (in.enable_nodal_pressure) {
    write_vtk_vectors(stream, "v_prime", s.v_prime);
  }
  stream.close();
}

}
