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
  host_s.time = s.time;

  auto const num_points = s.points.size();
  auto const num_nodes = s.nodes.size();

  host_s.x.resize(num_nodes);
  host_s.xp.resize(num_points);
  host_s.V.resize(num_points);
  host_s.rho.resize(num_points);

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
  host_s.potential_density.resize(num_points);
  hpc::copy(s.potential_density, host_s.potential_density);

  auto const Fp_size = s.Fp_total.size();
  if (Fp_size > 0) {
    host_s.Fp_total.resize(Fp_size);
    hpc::copy(s.Fp_total, host_s.Fp_total);
  }
  auto const ep_size = s.ep.size();
  if (ep_size > 0) {
    host_s.ep.resize(ep_size);
    hpc::copy(s.ep, host_s.ep);
  }
  auto const ep_dot_size = s.ep_dot.size();
  if (ep_dot_size > 0) {
    host_s.ep_dot.resize(ep_dot_size);
    hpc::copy(s.ep_dot, host_s.ep_dot);
  }
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
  write_vtk_scalars(point_stream, "potential_density", host_s.potential_density);
  if (host_s.Fp_total.size() > 0) {
    write_vtk_full_tensors(point_stream, "plastic_deformation_gradient", host_s.Fp_total);
  }
  if (host_s.ep.size() > 0) {
    write_vtk_scalars(point_stream, "ep", host_s.ep);
  }
  if (host_s.ep_dot.size() > 0) {
    write_vtk_scalars(point_stream, "ep_dot", host_s.ep_dot);
  }
  point_stream.close();
}

void otm_file_writer::to_console()
{
  std::cout << "TIME : " << host_s.time << '\n';
  auto const nodes_to_x = host_s.x.cbegin();
  auto print_x = [=] HPC_HOST (lgr::node_index const node) {
    auto const x = nodes_to_x[node].load();
    std::cout << "node: " << node << ", x:" << x << '\n';
  };
  hpc::for_each(hpc::host_policy(), host_s.nodes, print_x);

  auto const nodes_to_v = host_s.v.cbegin();
  auto print_v = [=] HPC_HOST (lgr::node_index const node) {
    auto const v = nodes_to_v[node].load();
    std::cout << "node: " << node << ", v:" << v << '\n';
  };
  hpc::for_each(hpc::host_policy(), host_s.nodes, print_v);

  auto const nodes_to_u = host_s.u.cbegin();
  auto print_u = [=] HPC_HOST (lgr::node_index const node) {
    auto const u = nodes_to_u[node].load();
    std::cout << "node: " << node << ", u:" << u << '\n';
  };
  hpc::for_each(hpc::host_policy(), host_s.nodes, print_u);

  auto const points_to_xp = host_s.xp.cbegin();
  auto print_xp = [=] HPC_HOST (lgr::point_index const point) {
    auto const xp = points_to_xp[point].load();
    std::cout << "point: " << point << ", xp:" << xp << '\n';
  };
  hpc::for_each(hpc::host_policy(), host_s.points, print_xp);

  auto const points_to_F = host_s.F_total.cbegin();
  auto print_F = [=] HPC_HOST (lgr::point_index const point) {
    auto const F = points_to_F[point].load();
    std::cout << "point: " << point << ", F:\n" << F;
  };
  hpc::for_each(hpc::host_policy(), host_s.points, print_F);

  auto const points_to_sigma = host_s.sigma.cbegin();
  auto const points_to_K = host_s.K.cbegin();
  auto const points_to_G = host_s.G.cbegin();
  auto print_sigma = [=] HPC_HOST (lgr::point_index const point) {
    auto const sigma = points_to_sigma[point].load();
    auto const K = points_to_K[point];
    auto const G = points_to_G[point];
    std::cout << "point: " << point << ", K: " << K << ", G: " << G << ", sigma:\n" << sigma;
  };
  hpc::for_each(hpc::host_policy(), host_s.points, print_sigma);

}

}
