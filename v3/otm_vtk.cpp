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

  host_s.v.resize(num_nodes);
  hpc::copy(s.v, host_s.v);
  host_s.u.resize(num_nodes);
  hpc::copy(s.u, host_s.u);
  host_s.mass.resize(num_nodes);
  hpc::copy(s.mass, host_s.mass);

  host_s.F_total.resize(num_points);
  hpc::copy(s.F_total, host_s.F_total);
  host_s.sigma.resize(num_points);
  hpc::copy(s.sigma_full, host_s.sigma);
  host_s.G.resize(num_points);
  hpc::copy(s.G, host_s.G);
  host_s.K.resize(num_points);
  hpc::copy(s.K, host_s.K);
}

void otm_file_writer::write(int const file_output_index) {
  auto node_stream = make_vtk_output_stream(prefix + "_nodes", file_output_index);
  auto point_stream = make_vtk_output_stream(prefix + "_points", file_output_index);

  start_vtk_unstructured_grid_file(node_stream);
  //POINTS (nodes)
  write_vtk_points(node_stream, host_s.x);
  write_vtk_point_data(node_stream, host_s.nodes);
  assert(host_s.x.size() == host_s.nodes.size());
  write_vtk_vectors(node_stream, "node_position", host_s.x);
  write_vtk_vectors(node_stream, "node_displacement", host_s.u);
  write_vtk_vectors(node_stream, "node_velocity", host_s.v);
  write_vtk_scalars(node_stream, "node_mass", host_s.mass);
  node_stream.close();

  start_vtk_unstructured_grid_file(point_stream);
  //POINTS (points)
  write_vtk_points(point_stream, host_s.xp);
  write_vtk_point_data(point_stream, host_s.points);
  assert(host_s.xp.size() == host_s.points.size());
  write_vtk_vectors(point_stream, "point_position", host_s.xp);
  write_vtk_scalars(point_stream, "density", host_s.rho);
  write_vtk_scalars(point_stream, "volume", host_s.V);
  write_vtk_full_tensors(point_stream, "sigma", host_s.sigma);
  write_vtk_full_tensors(point_stream, "deformation_gradient", host_s.F_total);
  write_vtk_scalars(point_stream, "G", host_s.G);
  write_vtk_scalars(point_stream, "K", host_s.K);
  point_stream.close();
}

}
