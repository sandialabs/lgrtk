#include <gtest/gtest.h>
#include <hpc_algorithm.hpp>
#include <hpc_array_traits.hpp>
#include <hpc_array_vector.hpp>
#include <hpc_dimensional.hpp>
#include <hpc_execution.hpp>
#include <hpc_functional.hpp>
#include <hpc_macros.hpp>
#include <hpc_numeric.hpp>
#include <hpc_range.hpp>
#include <hpc_vector.hpp>
#include <hpc_vector3.hpp>
#include <lgr_mesh_indices.hpp>
#include <lgr_state.hpp>
#include <otm_adapt.hpp>
#include <otm_util.hpp>
#include <unit_tests/otm_unit_mesh.hpp>

TEST(map, maxent_populate_nodes)
{
  lgr::state s;
  hexahedron_eight_points(s);
  hpc::pinned_array_vector<hpc::position<double>, lgr::node_index> x;
  hpc::pinned_array_vector<hpc::position<double>, lgr::node_index> u;
  hpc::pinned_array_vector<hpc::position<double>, lgr::node_index> v;
  auto const num_nodes_old = s.nodes.size();
  auto const num_nodes_new = num_nodes_old + 1;
  x.resize(num_nodes_new);
  u.resize(num_nodes_new);
  v.resize(num_nodes_new);
  hpc::copy(s.x, x);
  x[num_nodes_new - 1] = hpc::position<double>(0,  0,  0);
  auto const u0 = hpc::position<double>(12.0e-06,  4.0e-06,  3.0e-06);
  auto const v0 = hpc::velocity<double>(3.0, 4.0, 12.0);
  hpc::fill(hpc::host_policy(), u, u0);
  hpc::fill(hpc::host_policy(), v, v0);
  u[num_nodes_new - 1] = hpc::position<double>(0,  0,  0);
  v[num_nodes_new - 1] = hpc::velocity<double>(0,  0,  0);
  s.x.resize(num_nodes_new);
  s.u.resize(num_nodes_new);
  s.v.resize(num_nodes_new);
  hpc::copy(x, s.x);
  hpc::copy(u, s.u);
  hpc::copy(v, s.v);
  otm_populate_new_nodes(s, 0, num_nodes_old, num_nodes_old, num_nodes_new);
  hpc::copy(s.u, u);
  hpc::copy(s.v, v);
  auto const error_u = std::abs(hpc::norm(u[num_nodes_new - 1].load()) / 13.0e-06 - 1.0);
  auto const error_v = std::abs(hpc::norm(v[num_nodes_new - 1].load()) / 13.0 - 1.0);
  auto const eps = hpc::machine_epsilon<double>();
  ASSERT_LE(error_u, eps);
  ASSERT_LE(error_v, eps);
}

TEST(map, maxent_populate_points)
{
  lgr::state s;
  hexahedron_eight_points(s);
  hpc::pinned_array_vector<hpc::position<double>, lgr::point_index> xp;
  hpc::pinned_vector<hpc::pressure<double>, lgr::point_index> K;
  hpc::pinned_vector<hpc::pressure<double>, lgr::point_index> G;
  hpc::pinned_vector<hpc::density<double>, lgr::point_index> rho;
  hpc::pinned_vector<hpc::strain<double>, lgr::point_index> ep;
  hpc::pinned_vector<hpc::strain_rate<double>, lgr::point_index> ep_dot;
  hpc::pinned_array_vector<hpc::acceleration<double>, lgr::point_index> b;
  hpc::pinned_vector<hpc::volume<double>, lgr::point_index> V;
  auto const num_points_old = s.points.size();
  auto const num_points_new = num_points_old + 1;
  xp.resize(num_points_new);
  K.resize(num_points_new);
  G.resize(num_points_new);
  rho.resize(num_points_new);
  ep.resize(num_points_new);
  ep_dot.resize(num_points_new);
  b.resize(num_points_new);
  V.resize(num_points_new);
  hpc::copy(s.xp, xp);
  xp[num_points_new - 1] = hpc::position<double>(0,  0,  0);
  auto const K0 = hpc::pressure<double>(2.0e+09);
  auto const G0 = hpc::pressure<double>(1.0e+09);
  auto const rho0 = hpc::density<double>(1.0e+03);
  auto const ep0 = hpc::strain<double>(1.0e-01);
  auto const ep_dot0 = hpc::strain_rate<double>(1.0e+01);
  auto const b0 = hpc::acceleration<double>(0.0, 0.0, -9.81);
  auto const V0 = hpc::reduce(hpc::device_policy(), s.V, hpc::volume<double>(0.0));
  hpc::fill(hpc::host_policy(), K, K0);
  hpc::fill(hpc::host_policy(), G, G0);
  hpc::fill(hpc::host_policy(), rho, rho0);
  hpc::fill(hpc::host_policy(), ep, ep0);
  hpc::fill(hpc::host_policy(), ep_dot, ep_dot0);
  hpc::fill(hpc::host_policy(), b, b0);
  hpc::copy(s.V, V);
  K[num_points_new - 1] = hpc::pressure<double>(0.0);
  G[num_points_new - 1] = hpc::pressure<double>(0.0);
  rho[num_points_new - 1] = hpc::density<double>(0.0);
  ep[num_points_new - 1] = hpc::strain<double>(0.0);
  ep_dot[num_points_new - 1] = hpc::strain_rate<double>(0.0);
  b[num_points_new - 1] = hpc::acceleration<double>(0.0, 0.0, 0.0);
  V[num_points_new - 1] = hpc::volume<double>(0.0);
  s.xp.resize(num_points_new);
  s.K.resize(num_points_new);
  s.G.resize(num_points_new);
  s.rho.resize(num_points_new);
  s.ep.resize(num_points_new);
  s.ep_dot.resize(num_points_new);
  s.b.resize(num_points_new);
  s.V.resize(num_points_new);
  hpc::copy(xp, s.xp);
  hpc::copy(K, s.K);
  hpc::copy(G, s.G);
  hpc::copy(rho, s.rho);
  hpc::copy(ep, s.ep);
  hpc::copy(ep_dot, s.ep_dot);
  hpc::copy(b, s.b);
  hpc::copy(V, s.V);
  otm_populate_new_points(s, 0, num_points_old, num_points_old, num_points_new);
  hpc::copy(s.K, K);
  hpc::copy(s.G, G);
  hpc::copy(s.rho, rho);
  hpc::copy(s.ep, ep);
  hpc::copy(s.ep_dot, ep_dot);
  hpc::copy(s.b, b);
  hpc::copy(s.V, V);
  auto const error_K = std::abs(K[num_points_new - 1] / K0 - 1.0);
  auto const error_G = std::abs(G[num_points_new - 1] / G0 - 1.0);
  auto const error_rho = std::abs(rho[num_points_new - 1] / rho0 - 1.0);
  auto const error_ep = std::abs(ep[num_points_new - 1] / ep0 - 1.0);
  auto const error_ep_dot = std::abs(ep_dot[num_points_new - 1] / ep_dot0 - 1.0);
  auto const error_b = hpc::norm(b[num_points_new - 1].load() - b0) / hpc::norm(b0);
  auto const eps = hpc::machine_epsilon<double>();
  auto const error_V = std::abs(num_points_new * V[num_points_new - 1] / V0 - 1.0);
  ASSERT_LE(error_K, eps);
  ASSERT_LE(error_G, eps);
  ASSERT_LE(error_rho, eps);
  ASSERT_LE(error_ep, eps);
  ASSERT_LE(error_ep_dot, eps);
  ASSERT_LE(error_b, eps);
  ASSERT_LE(error_V, eps);
}
