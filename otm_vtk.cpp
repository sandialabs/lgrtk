#include <hpc_array_vector.hpp>
#include <hpc_range.hpp>
#include <hpc_vector.hpp>
#include <lgr_state.hpp>
#include <lgr_vtk_util.hpp>
#include <otm_vtk.hpp>
#include <cassert>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace lgr {

void otm_file_writer::capture(state const& s) {
  host_s.nodes = s.nodes;
  host_s.points = s.points;

  auto num_points = s.points.size();
  auto num_nodes = s.nodes.size();
  lgr::resize_state_arrays(host_s, num_nodes, num_points);

  hpc::copy(s.x, host_s.x);
  hpc::copy(s.xp, host_s.xp);

  hpc::copy(s.rho, host_s.rho);
  hpc::copy(s.V, host_s.V);
}

void otm_file_writer::write(int const file_output_index) {
  auto node_stream = make_vtk_output_stream(prefix + "_nodes", file_output_index);
  auto point_stream = make_vtk_output_stream(prefix + "_points", file_output_index);

  node_stream << std::scientific << std::setprecision(17);
  start_vtk_unstructured_grid_file(node_stream);
  //POINTS (nodes)
  write_vtk_points(node_stream, host_s.x);
  write_vtk_point_data(node_stream, host_s.nodes);
  assert(host_s.x.size() == host_s.nodes.size());
  write_vtk_vectors(node_stream, "node_position", host_s.x);
  node_stream.close();

  start_vtk_unstructured_grid_file(point_stream);
  //POINTS (points)
  write_vtk_points(point_stream, host_s.xp);
  write_vtk_point_data(point_stream, host_s.points);
  assert(host_s.xp.size() == host_s.points.size());
  write_vtk_vectors(point_stream, "point_position", host_s.xp);
  write_vtk_scalars(point_stream, "density", host_s.rho);
  write_vtk_scalars(point_stream, "volume", host_s.V);
  point_stream.close();
}

}
