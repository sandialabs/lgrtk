#pragma once

#include <hpc_array_vector.hpp>
#include <hpc_dimensional.hpp>
#include <hpc_range.hpp>
#include <hpc_vector.hpp>
#include <hpc_vector3.hpp>
#include <lgr_input.hpp>
#include <lgr_mesh_indices.hpp>
#include <lgr_state.hpp>
#include <otm_meshless.hpp>

inline void
tetrahedron_single_point(lgr::state& s)
{
  using NI = lgr::node_index;
  auto const num_nodes = NI(4);
  s.nodes.resize(num_nodes);
  s.x.resize(s.nodes.size());
  auto const nodes_to_x = s.x.begin();
  nodes_to_x[NI(0)] = hpc::position<double>(0, 0, 0);
  nodes_to_x[NI(1)] = hpc::position<double>(1, 0, 0);
  nodes_to_x[NI(2)] = hpc::position<double>(0, 1, 0);
  nodes_to_x[NI(3)] = hpc::position<double>(0, 0, 1);

  using PI = lgr::point_index;
  auto const num_points = PI(1);
  s.xp.resize(num_points);
  auto const points_to_xp = s.xp.begin();
  points_to_xp[PI(0)] = hpc::position<double>(0.25, 0.25, 0.25);

  using PNI = lgr::point_node_index;
  s.N.resize(PNI(num_points * num_nodes));
  s.grad_N.resize(PNI(num_points * num_nodes));

  s.h_otm.resize(num_points);
  auto const points_to_h = s.h_otm.begin();
  points_to_h[PI(0)] = 1.0;

  s.points.resize(num_points);

  hpc::device_vector<PNI, PI> support_sizes(num_points, PNI(num_nodes));
  s.points_to_point_nodes.assign_sizes(support_sizes);

  s.point_nodes_to_nodes.resize(num_points * num_nodes);
  auto const point_nodes_to_nodes = s.point_nodes_to_nodes.begin();
  point_nodes_to_nodes[PNI(0)] = NI(0);
  point_nodes_to_nodes[PNI(1)] = NI(1);
  point_nodes_to_nodes[PNI(2)] = NI(2);
  point_nodes_to_nodes[PNI(3)] = NI(3);

  using NPI = lgr::node_point_index;
  hpc::device_vector<NPI, NI> influence_sizes(num_nodes, NPI(num_points));
  s.nodes_to_node_points.assign_sizes(influence_sizes);

  s.node_points_to_points.resize(num_nodes * num_points);
  auto const node_points_to_points = s.node_points_to_points.begin();
  node_points_to_points[NPI(0)] = PI(0);
  node_points_to_points[NPI(1)] = PI(0);
  node_points_to_points[NPI(2)] = PI(0);
  node_points_to_points[NPI(3)] = PI(0);

  s.node_points_to_point_nodes.resize(num_nodes * num_points);
  auto const node_points_to_point_nodes = s.node_points_to_point_nodes.begin();
  node_points_to_point_nodes[NPI(0)] = PNI(0);
  node_points_to_point_nodes[NPI(1)] = PNI(1);
  node_points_to_point_nodes[NPI(2)] = PNI(2);
  node_points_to_point_nodes[NPI(3)] = PNI(3);

  lgr::otm_initialize_grad_val_N(s);

  s.mass.resize(num_nodes);
  s.V.resize(num_points);
  s.rho.resize(num_points);

  auto const V = s.V.begin();
  auto const rho = s.rho.begin();

  V[PI(0)] = 1.0 / 6.0;
  rho[PI(0)] = 1000.0;
}

inline void
two_tetrahedra_two_points(lgr::state& s)
{
  using NI = lgr::node_index;
  auto const num_nodes = NI(5);
  s.nodes.resize(num_nodes);
  s.x.resize(s.nodes.size());
  auto const nodes_to_x = s.x.begin();
  nodes_to_x[NI(0)] = hpc::position<double>(0, 0, 0);
  nodes_to_x[NI(1)] = hpc::position<double>(1, 0, 0);
  nodes_to_x[NI(2)] = hpc::position<double>(0, 1, 0);
  nodes_to_x[NI(3)] = hpc::position<double>(0, 0, 1);
  nodes_to_x[NI(4)] = hpc::position<double>(1, 1, 1);

  using PI = lgr::point_index;
  auto const num_points = PI(2);
  s.xp.resize(num_points);
  auto const points_to_xp = s.xp.begin();
  points_to_xp[PI(0)] = hpc::position<double>(0.25, 0.25, 0.25);
  points_to_xp[PI(1)] = hpc::position<double>(0.50, 0.50, 0.50);

  using PNI = lgr::point_node_index;
  s.N.resize(PNI(num_points * num_nodes));
  s.grad_N.resize(PNI(num_points * num_nodes));

  s.h_otm.resize(num_points);
  auto const points_to_h = s.h_otm.begin();
  points_to_h[PI(0)] = 1.0;
  points_to_h[PI(1)] = 1.0;

  s.points.resize(num_points);

  hpc::device_vector<PNI, PI> support_sizes(num_points, PNI(num_nodes));
  s.points_to_point_nodes.assign_sizes(support_sizes);

  s.point_nodes_to_nodes.resize(num_points * num_nodes);
  auto const point_nodes_to_nodes = s.point_nodes_to_nodes.begin();
  point_nodes_to_nodes[PNI(0)] = NI(0);
  point_nodes_to_nodes[PNI(1)] = NI(1);
  point_nodes_to_nodes[PNI(2)] = NI(2);
  point_nodes_to_nodes[PNI(3)] = NI(3);
  point_nodes_to_nodes[PNI(4)] = NI(4);

  point_nodes_to_nodes[PNI(5)] = NI(0);
  point_nodes_to_nodes[PNI(6)] = NI(1);
  point_nodes_to_nodes[PNI(7)] = NI(2);
  point_nodes_to_nodes[PNI(8)] = NI(3);
  point_nodes_to_nodes[PNI(9)] = NI(4);

  using NPI = lgr::node_point_index;
  hpc::device_vector<NPI, NI> influence_sizes(num_nodes, NPI(num_points));
  s.nodes_to_node_points.assign_sizes(influence_sizes);

  s.node_points_to_points.resize(num_nodes * num_points);
  auto const node_points_to_points = s.node_points_to_points.begin();
  node_points_to_points[NPI(0)] = PI(0);
  node_points_to_points[NPI(1)] = PI(1);
  node_points_to_points[NPI(2)] = PI(0);
  node_points_to_points[NPI(3)] = PI(1);
  node_points_to_points[NPI(4)] = PI(0);
  node_points_to_points[NPI(5)] = PI(1);
  node_points_to_points[NPI(6)] = PI(0);
  node_points_to_points[NPI(7)] = PI(1);
  node_points_to_points[NPI(8)] = PI(0);
  node_points_to_points[NPI(9)] = PI(1);

  s.node_points_to_point_nodes.resize(num_nodes * num_points);
  auto const node_points_to_point_nodes = s.node_points_to_point_nodes.begin();
  node_points_to_point_nodes[NPI(0)] = PNI(0);
  node_points_to_point_nodes[NPI(1)] = PNI(0);
  node_points_to_point_nodes[NPI(2)] = PNI(1);
  node_points_to_point_nodes[NPI(3)] = PNI(1);
  node_points_to_point_nodes[NPI(4)] = PNI(2);
  node_points_to_point_nodes[NPI(5)] = PNI(3);
  node_points_to_point_nodes[NPI(6)] = PNI(3);
  node_points_to_point_nodes[NPI(7)] = PNI(3);
  node_points_to_point_nodes[NPI(8)] = PNI(4);
  node_points_to_point_nodes[NPI(9)] = PNI(4);

  lgr::otm_initialize_grad_val_N(s);

  s.mass.resize(num_nodes);
  s.V.resize(num_points);
  s.rho.resize(num_points);

  auto const V = s.V.begin();
  auto const rho = s.rho.begin();

  V[PI(0)] = 1.0 / 6.0;
  rho[PI(0)] = 1000.0;

  auto const a = nodes_to_x[NI(1)].load();
  auto const b = nodes_to_x[NI(2)].load();
  auto const c = nodes_to_x[NI(3)].load();
  auto const d = nodes_to_x[NI(4)].load();
  auto const ad = a - d;
  auto const bd = b - d;
  auto const cd = c - d;

  V[PI(1)] = std::abs(hpc::inner_product(ad, hpc::cross(bd, cd))) / 6.0;
  rho[PI(1)] = 1000.0;
}

inline void
hexahedron_eight_points(lgr::state& s)
{
  using NI = lgr::node_index;
  auto const num_nodes = NI(8);
  s.nodes.resize(num_nodes);
  s.x.resize(s.nodes.size());
  auto const nodes_to_x = s.x.begin();
  nodes_to_x[NI(0)] = hpc::position<double>(-1, -1, -1);
  nodes_to_x[NI(1)] = hpc::position<double>( 1, -1, -1);
  nodes_to_x[NI(2)] = hpc::position<double>( 1,  1, -1);
  nodes_to_x[NI(3)] = hpc::position<double>(-1,  1, -1);
  nodes_to_x[NI(4)] = hpc::position<double>(-1, -1,  1);
  nodes_to_x[NI(5)] = hpc::position<double>( 1, -1,  1);
  nodes_to_x[NI(6)] = hpc::position<double>( 1,  1,  1);
  nodes_to_x[NI(7)] = hpc::position<double>(-1,  1,  1);

  using PI = lgr::point_index;
  auto const num_points = PI(8);
  s.xp.resize(num_points);
  auto const points_to_xp = s.xp.begin();
  auto const g = std::sqrt(3.0) / 3.0;
  points_to_xp[PI(0)] = hpc::position<double>(-g, -g, -g);
  points_to_xp[PI(1)] = hpc::position<double>( g, -g, -g);
  points_to_xp[PI(2)] = hpc::position<double>( g,  g, -g);
  points_to_xp[PI(3)] = hpc::position<double>(-g,  g, -g);
  points_to_xp[PI(4)] = hpc::position<double>(-g, -g,  g);
  points_to_xp[PI(5)] = hpc::position<double>( g, -g,  g);
  points_to_xp[PI(6)] = hpc::position<double>( g,  g,  g);
  points_to_xp[PI(7)] = hpc::position<double>(-g,  g,  g);

  using PNI = lgr::point_node_index;
  s.N.resize(PNI(num_points * num_nodes));
  s.grad_N.resize(PNI(num_points * num_nodes));

  s.h_otm.resize(num_points);
  auto const points_to_h = s.h_otm.begin();
  for (auto i = 0; i < num_points; ++i) {
    points_to_h[PI(i)] = 2.0;
  }

  s.points.resize(num_points);

  hpc::device_vector<PNI, PI> support_sizes(num_points, PNI(num_nodes));
  s.points_to_point_nodes.assign_sizes(support_sizes);

  s.point_nodes_to_nodes.resize(num_points * num_nodes);
  auto const point_nodes_to_nodes = s.point_nodes_to_nodes.begin();
  for (auto i = 0; i < num_points * num_nodes; ++i) {
    auto const point_node = i % num_points;
    point_nodes_to_nodes[PNI(i)] = NI(point_node);

  }

  using NPI = lgr::node_point_index;
  hpc::device_vector<NPI, NI> influence_sizes(num_nodes, NPI(num_points));
  s.nodes_to_node_points.assign_sizes(influence_sizes);

  s.node_points_to_points.resize(num_nodes * num_points);
  auto const node_points_to_points = s.node_points_to_points.begin();
  for (auto i = 0; i < num_points * num_nodes; ++i) {
    auto const node_point = i % num_nodes;
    node_points_to_points[NPI(i)] = PI(node_point);

  }

  s.node_points_to_point_nodes.resize(num_nodes * num_points);
  auto const node_points_to_point_nodes = s.node_points_to_point_nodes.begin();
  for (auto i = 0; i < num_points * num_nodes; ++i) {
    auto const node_ordinal = i / num_points;
    node_points_to_point_nodes[NPI(i)] = PNI(node_ordinal);
  }

  lgr::otm_initialize_grad_val_N(s);

  s.mass.resize(num_nodes);
  s.V.resize(num_points);
  s.rho.resize(num_points);

  auto const V = s.V.begin();
  auto const rho = s.rho.begin();

  for (auto i = 0; i < num_points; ++i) {
    V[PI(i)] = 1.0;
    rho[PI(i)] = 1000.0;
  }

}
