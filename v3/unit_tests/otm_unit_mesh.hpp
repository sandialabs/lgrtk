#pragma once

#include <hpc_array_vector.hpp>
#include <hpc_dimensional.hpp>
#include <hpc_range.hpp>
#include <hpc_vector.hpp>
#include <hpc_vector3.hpp>
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
  s.xm.resize(num_points);
  auto const points_to_xm = s.xm.begin();
  points_to_xm[PI(0)] = hpc::position<double>(0.25, 0.25, 0.25);

  using PNI = lgr::point_node_index;
  s.N.resize(PNI(num_points * num_nodes));
  s.grad_N.resize(PNI(num_points * num_nodes));

  s.h_otm.resize(num_points);
  auto const points_to_h = s.h_otm.begin();
  points_to_h[PI(0)] = 1.0;

  s.points.resize(num_points);

  using NSI = lgr::node_in_support_index;
  hpc::device_vector<NSI, PI> support_sizes(num_points, NSI(num_nodes));
  s.nodes_in_support.assign_sizes(support_sizes);

  s.points_to_supported_nodes.resize(num_points * num_nodes);
  auto const support_nodes_to_nodes = s.points_to_supported_nodes.begin();
  support_nodes_to_nodes[PNI(0)] = NI(0);
  support_nodes_to_nodes[PNI(1)] = NI(1);
  support_nodes_to_nodes[PNI(2)] = NI(2);
  support_nodes_to_nodes[PNI(3)] = NI(3);

  using PII = lgr::point_in_influence_index;
  hpc::device_vector<PII, NI> influence_sizes(num_nodes, PII(num_points));
  s.points_in_influence.assign_sizes(influence_sizes);

  using NPI = lgr::node_point_index;
  s.nodes_to_influenced_points.resize(num_nodes * num_points);
  auto const influence_points_to_points = s.nodes_to_influenced_points.begin();
  influence_points_to_points[NPI(0)] = PI(0);
  influence_points_to_points[NPI(1)] = PI(0);
  influence_points_to_points[NPI(2)] = PI(0);
  influence_points_to_points[NPI(3)] = PI(0);

  lgr::initialize_meshless_grad_val_N(s);

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
  s.xm.resize(num_points);
  auto const points_to_xm = s.xm.begin();
  points_to_xm[PI(0)] = hpc::position<double>(0.25, 0.25, 0.25);
  points_to_xm[PI(1)] = hpc::position<double>(0.75, 0.75, 0.75);

  using PNI = lgr::point_node_index;
  s.N.resize(PNI(num_points * num_nodes));
  s.grad_N.resize(PNI(num_points * num_nodes));

  s.h_otm.resize(num_points);
  auto const points_to_h = s.h_otm.begin();
  points_to_h[PI(0)] = 1.0;
  points_to_h[PI(1)] = 1.0;

  s.points.resize(num_points);

  using NSI = lgr::node_in_support_index;
  hpc::device_vector<lgr::node_in_support_index, lgr::point_index> support_sizes(s.points.size(), NSI(num_nodes));
  s.nodes_in_support.assign_sizes(support_sizes);

  s.points_to_supported_nodes.resize(num_points * num_nodes);
  auto const support_nodes_to_nodes = s.points_to_supported_nodes.begin();
  support_nodes_to_nodes[PNI(0)] = NI(0);
  support_nodes_to_nodes[PNI(1)] = NI(1);
  support_nodes_to_nodes[PNI(2)] = NI(2);
  support_nodes_to_nodes[PNI(3)] = NI(3);
  support_nodes_to_nodes[PNI(4)] = NI(4);

  support_nodes_to_nodes[PNI(5)] = NI(0);
  support_nodes_to_nodes[PNI(6)] = NI(1);
  support_nodes_to_nodes[PNI(7)] = NI(2);
  support_nodes_to_nodes[PNI(8)] = NI(3);
  support_nodes_to_nodes[PNI(9)] = NI(4);

  lgr::initialize_meshless_grad_val_N(s);
}
