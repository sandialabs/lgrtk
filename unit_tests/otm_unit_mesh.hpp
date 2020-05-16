#pragma once

#include <hpc_algorithm.hpp>
#include <hpc_array_vector.hpp>
#include <hpc_dimensional.hpp>
#include <hpc_execution.hpp>
#include <hpc_macros.hpp>
#include <hpc_range.hpp>
#include <hpc_range_sum.hpp>
#include <hpc_vector.hpp>
#include <hpc_vector3.hpp>
#include <lgr_domain.hpp>
#include <lgr_input.hpp>
#include <lgr_mesh_indices.hpp>
#include <lgr_meshing.hpp>
#include <lgr_state.hpp>
#include <otm_host_pinned_state.hpp>
#include <otm_meshless.hpp>
#include <otm_tet2meshless.hpp>
#include <otm_tetrahedron_util.hpp>
#include <algorithm>
#include <hpc_math.hpp>
#include <functional>
#include <memory>
#include <string>
#include <vector>

inline void
tetrahedron_single_point(lgr::state& s)
{
  using NI = lgr::node_index;
  using PI = lgr::point_index;
  auto const num_nodes = NI(4);
  auto const num_points = PI(1);

  lgr::otm_host_pinned_state host_s;

  lgr::resize_state_arrays(s, num_nodes, num_points);
  lgr::resize_state_arrays(host_s, num_nodes, num_points);

  s.nodes.resize(num_nodes);
  auto const nodes_to_x = host_s.x.begin();
  nodes_to_x[NI(0)] = hpc::position<double>(0, 0, 0);
  nodes_to_x[NI(1)] = hpc::position<double>(1, 0, 0);
  nodes_to_x[NI(2)] = hpc::position<double>(0, 1, 0);
  nodes_to_x[NI(3)] = hpc::position<double>(0, 0, 1);
  hpc::copy(host_s.x, s.x);

  auto const points_to_xp = host_s.xp.begin();
  points_to_xp[PI(0)] = hpc::position<double>(0.25, 0.25, 0.25);
  hpc::copy(host_s.xp, s.xp);

  using PNI = lgr::point_node_index;
  s.N.resize(PNI(num_points * num_nodes));
  s.grad_N.resize(PNI(num_points * num_nodes));

  s.h_otm.resize(num_points);
  hpc::fill(hpc::device_policy(), s.h_otm, 1.0);

  s.points.resize(num_points);
  hpc::device_vector<PNI, PI> support_sizes(num_points, PNI(num_nodes));
  s.points_to_point_nodes.assign_sizes(support_sizes);

  auto const point_nodes_to_nodes = host_s.point_nodes_to_nodes.begin();
  point_nodes_to_nodes[PNI(0)] = NI(0);
  point_nodes_to_nodes[PNI(1)] = NI(1);
  point_nodes_to_nodes[PNI(2)] = NI(2);
  point_nodes_to_nodes[PNI(3)] = NI(3);
  hpc::copy(host_s.point_nodes_to_nodes, s.point_nodes_to_nodes);

  using NPI = lgr::node_point_index;
  hpc::device_vector<NPI, NI> influence_sizes(num_nodes, NPI(num_points));
  s.nodes_to_node_points.assign_sizes(influence_sizes);

  auto const node_points_to_points = host_s.node_points_to_points.begin();
  node_points_to_points[NPI(0)] = PI(0);
  node_points_to_points[NPI(1)] = PI(0);
  node_points_to_points[NPI(2)] = PI(0);
  node_points_to_points[NPI(3)] = PI(0);
  hpc::copy(host_s.node_points_to_points, s.node_points_to_points);

  auto const node_points_to_point_nodes = host_s.node_points_to_point_nodes.begin();
  node_points_to_point_nodes[NPI(0)] = PNI(0);
  node_points_to_point_nodes[NPI(1)] = PNI(1);
  node_points_to_point_nodes[NPI(2)] = PNI(2);
  node_points_to_point_nodes[NPI(3)] = PNI(3);
  hpc::copy(host_s.node_points_to_point_nodes, s.node_points_to_point_nodes);

  double const gamma_otm = 1.5;
  double const h = 1.0;
  s.otm_beta = gamma_otm / (h * h);

  lgr::otm_update_shape_functions(s);

  s.mass.resize(num_nodes);

  hpc::fill(hpc::device_policy(), s.V, 1.0/6.0);
  hpc::fill(hpc::device_policy(), s.rho, 1000.0);
}

inline void
two_tetrahedra_two_points(lgr::state& s)
{
  using NI = lgr::node_index;
  using PI = lgr::point_index;
  auto const num_nodes = NI(5);
  auto const num_points = PI(2);

  lgr::otm_host_pinned_state host_s;

  lgr::resize_state_arrays(s, num_nodes, num_points);
  lgr::resize_state_arrays(host_s, num_nodes, num_points);

  s.nodes.resize(num_nodes);
  auto const nodes_to_x = host_s.x.begin();
  nodes_to_x[NI(0)] = hpc::position<double>(0, 0, 0);
  nodes_to_x[NI(1)] = hpc::position<double>(1, 0, 0);
  nodes_to_x[NI(2)] = hpc::position<double>(0, 1, 0);
  nodes_to_x[NI(3)] = hpc::position<double>(0, 0, 1);
  nodes_to_x[NI(4)] = hpc::position<double>(1, 1, 1);
  hpc::copy(host_s.x, s.x);

  auto const points_to_xp = host_s.xp.begin();
  points_to_xp[PI(0)] = hpc::position<double>(0.25, 0.25, 0.25);
  points_to_xp[PI(1)] = hpc::position<double>(0.50, 0.50, 0.50);
  hpc::copy(host_s.xp, s.xp);

  using PNI = lgr::point_node_index;
  s.N.resize(PNI(num_points * num_nodes));
  s.grad_N.resize(PNI(num_points * num_nodes));

  s.h_otm.resize(num_points);
  hpc::fill(hpc::device_policy(), s.h_otm, 1.0);

  s.points.resize(num_points);
  hpc::device_vector<PNI, PI> support_sizes(num_points, PNI(num_nodes));
  s.points_to_point_nodes.assign_sizes(support_sizes);

  auto const point_nodes_to_nodes = host_s.point_nodes_to_nodes.begin();
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
  hpc::copy(host_s.point_nodes_to_nodes, s.point_nodes_to_nodes);

  using NPI = lgr::node_point_index;
  hpc::device_vector<NPI, NI> influence_sizes(num_nodes, NPI(num_points));
  s.nodes_to_node_points.assign_sizes(influence_sizes);

  auto const node_points_to_points = host_s.node_points_to_points.begin();
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
  hpc::copy(host_s.node_points_to_points, s.node_points_to_points);

  auto const node_points_to_point_nodes = host_s.node_points_to_point_nodes.begin();
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
  hpc::copy(host_s.node_points_to_point_nodes, s.node_points_to_point_nodes);

  double const gamma_otm = 1.5;
  double const h = 1.0;
  s.otm_beta = gamma_otm / (h * h);

  lgr::otm_update_shape_functions(s);

  s.mass.resize(num_nodes);

  auto const V = host_s.V.begin();
  auto const rho = host_s.rho.begin();

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

  hpc::copy(host_s.V, s.V);
  hpc::copy(host_s.rho, s.rho);
}

inline void
hexahedron_eight_points(lgr::state& s)
{
  using NI = lgr::node_index;
  using PI = lgr::point_index;
  auto const num_nodes = NI(8);
  auto const num_points = PI(8);

  lgr::otm_host_pinned_state host_s;

  lgr::resize_state_arrays(s, num_nodes, num_points);
  lgr::resize_state_arrays(host_s, num_nodes, num_points);

  s.nodes.resize(num_nodes);
  auto const nodes_to_x = host_s.x.begin();
  nodes_to_x[NI(0)] = hpc::position<double>(-1, -1, -1);
  nodes_to_x[NI(1)] = hpc::position<double>( 1, -1, -1);
  nodes_to_x[NI(2)] = hpc::position<double>( 1,  1, -1);
  nodes_to_x[NI(3)] = hpc::position<double>(-1,  1, -1);
  nodes_to_x[NI(4)] = hpc::position<double>(-1, -1,  1);
  nodes_to_x[NI(5)] = hpc::position<double>( 1, -1,  1);
  nodes_to_x[NI(6)] = hpc::position<double>( 1,  1,  1);
  nodes_to_x[NI(7)] = hpc::position<double>(-1,  1,  1);
  hpc::copy(host_s.x, s.x);

  auto const points_to_xp = host_s.xp.begin();
  auto const g = std::sqrt(3.0) / 3.0;
  points_to_xp[PI(0)] = hpc::position<double>(-g, -g, -g);
  points_to_xp[PI(1)] = hpc::position<double>( g, -g, -g);
  points_to_xp[PI(2)] = hpc::position<double>( g,  g, -g);
  points_to_xp[PI(3)] = hpc::position<double>(-g,  g, -g);
  points_to_xp[PI(4)] = hpc::position<double>(-g, -g,  g);
  points_to_xp[PI(5)] = hpc::position<double>( g, -g,  g);
  points_to_xp[PI(6)] = hpc::position<double>( g,  g,  g);
  points_to_xp[PI(7)] = hpc::position<double>(-g,  g,  g);
  hpc::copy(host_s.xp, s.xp);

  using PNI = lgr::point_node_index;
  s.N.resize(PNI(num_points * num_nodes));
  s.grad_N.resize(PNI(num_points * num_nodes));

  s.h_otm.resize(num_points);
  hpc::fill(hpc::device_policy(), s.h_otm, 2.0);

  s.points.resize(num_points);
  hpc::device_vector<PNI, PI> support_sizes(num_points, PNI(num_nodes));
  s.points_to_point_nodes.assign_sizes(support_sizes);

  auto const point_nodes_to_nodes = host_s.point_nodes_to_nodes.begin();
  for (auto i = 0; i < num_points * num_nodes; ++i) {
    auto const point_node = i % num_points;
    point_nodes_to_nodes[PNI(i)] = NI(point_node);
  }
  hpc::copy(host_s.point_nodes_to_nodes, s.point_nodes_to_nodes);

  using NPI = lgr::node_point_index;
  hpc::device_vector<NPI, NI> influence_sizes(num_nodes, NPI(num_points));
  s.nodes_to_node_points.assign_sizes(influence_sizes);

  auto const node_points_to_points = host_s.node_points_to_points.begin();
  for (auto i = 0; i < num_points * num_nodes; ++i) {
    auto const node_point = i % num_nodes;
    node_points_to_points[NPI(i)] = PI(node_point);
  }
  hpc::copy(host_s.node_points_to_points, s.node_points_to_points);

  auto const node_points_to_point_nodes = host_s.node_points_to_point_nodes.begin();
  for (auto i = 0; i < num_points * num_nodes; ++i) {
    auto const node_ordinal = i / num_points;
    node_points_to_point_nodes[NPI(i)] = PNI(node_ordinal);
  }
  hpc::copy(host_s.node_points_to_point_nodes, s.node_points_to_point_nodes);

  lgr::otm_update_shape_functions(s);

  s.mass.resize(num_nodes);

  auto const V = host_s.V.begin();
  auto const rho = host_s.rho.begin();

  for (auto i = 0; i < num_points; ++i) {
    V[PI(i)] = 1.0;
    rho[PI(i)] = 1000.0;
  }
  hpc::copy(host_s.V, s.V);
  hpc::copy(host_s.rho, s.rho);
}

inline void compute_material_points_as_element_centroids(
    hpc::counting_range<lgr::point_index> const points,
    hpc::device_range_sum<lgr::point_node_index, lgr::point_index> const &points_to_point_nodes,
    hpc::device_vector<lgr::node_index, lgr::point_node_index> const &point_nodes_to_nodes,
    hpc::device_array_vector<hpc::position<double>, lgr::node_index> const &x,
    hpc::device_array_vector<hpc::position<double>, lgr::point_index> &xp)
{
  auto pt_to_pt_nodes = points_to_point_nodes.cbegin();
  auto pt_nodes_to_nodes = point_nodes_to_nodes.cbegin();
  auto x_nodes = x.cbegin();
  auto x_points = xp.begin();
  auto point_func = [=] HPC_DEVICE(const lgr::point_index point)
  {
    auto const point_nodes = pt_to_pt_nodes[point];
    hpc::position<double> avg_coord(0., 0., 0.);
    for (auto point_node : point_nodes)
    {
      auto const node = pt_nodes_to_nodes[point_node];
      avg_coord += x_nodes[node].load();
    }
    avg_coord /= point_nodes.size();

    x_points[point] = avg_coord;
  };
  hpc::for_each(hpc::device_policy(), points, point_func);
}

HPC_NOINLINE inline void elastic_wave_four_points_per_tetrahedron(lgr::state& s)
{
  using MI = lgr::material_index;
  constexpr MI x_boundary(1);
  constexpr MI y_boundary(2);
  constexpr MI z_boundary(3);
  lgr::input in(MI(1), MI(3));
  in.name = "elastic_wave_3d";
  in.element = lgr::TETRAHEDRON;
  in.elements_along_x = 1000;
  in.elements_along_y = 1;
  in.y_domain_size = 1.0e-3;
  in.elements_along_z = 1;
  in.z_domain_size = 1.0e-3;
  static constexpr hpc::vector3<double> x_axis(1.0, 0.0, 0.0);
  static constexpr hpc::vector3<double> y_axis(0.0, 1.0, 0.0);
  static constexpr hpc::vector3<double> z_axis(0.0, 0.0, 1.0);
  static constexpr double eps = 1.0e-10;
  auto x_domain = std::make_unique<lgr::union_domain>();
  x_domain->add(lgr::epsilon_around_plane_domain( { x_axis, 0.0 }, eps));
  x_domain->add(lgr::epsilon_around_plane_domain( { x_axis, in.x_domain_size }, eps));
  in.domains[x_boundary] = std::move(x_domain);
  in.zero_acceleration_conditions.push_back( { x_boundary, x_axis });
  auto y_domain = std::make_unique<lgr::union_domain>();
  y_domain->add(lgr::epsilon_around_plane_domain( { y_axis, 0.0 }, eps));
  y_domain->add(lgr::epsilon_around_plane_domain( { y_axis, in.y_domain_size }, eps));
  in.domains[y_boundary] = std::move(y_domain);
  in.zero_acceleration_conditions.push_back( { y_boundary, y_axis });
  auto z_domain = std::make_unique<lgr::union_domain>();
  z_domain->add(lgr::epsilon_around_plane_domain( { z_axis, 0.0 }, eps));
  z_domain->add(lgr::epsilon_around_plane_domain( { z_axis, in.z_domain_size }, eps));
  in.domains[z_boundary] = std::move(z_domain);
  in.zero_acceleration_conditions.push_back( { z_boundary, z_axis });

  build_mesh(in, s);
  if (in.x_transform)
    in.x_transform(&s.x);
  resize_state(in, s);

  auto const points_in_element = 4;
  using PEI = lgr::point_in_element_index;
  s.points_in_element.resize(PEI(points_in_element));
  in.otm_material_points_to_add_per_element = points_in_element;
  lgr::tet_nodes_to_points point_interpolator(points_in_element);
  in.xp_transform = std::ref(point_interpolator);

  convert_tet_mesh_to_meshless(in, s);
}
