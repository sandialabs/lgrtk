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
  auto const num_points_old = s.points.size();
  auto const num_points_new = num_points_old + 1;
  xp.resize(num_points_new);
  K.resize(num_points_new);
  G.resize(num_points_new);
  hpc::copy(s.xp, xp);
  xp[num_points_new - 1] = hpc::position<double>(0,  0,  0);
  auto const K0 = hpc::pressure<double>(2.0e+09);
  auto const G0 = hpc::pressure<double>(1.0e+09);
  hpc::fill(hpc::host_policy(), K, K0);
  hpc::fill(hpc::host_policy(), G, G0);
  K[num_points_new - 1] = hpc::pressure<double>(0.0);
  G[num_points_new - 1] = hpc::pressure<double>(0.0);
  s.xp.resize(num_points_new);
  s.K.resize(num_points_new);
  s.G.resize(num_points_new);
  hpc::copy(xp, s.xp);
  hpc::copy(K, s.K);
  hpc::copy(G, s.G);
  otm_populate_new_points(s, 0, num_points_old, num_points_old, num_points_new);
  hpc::copy(s.K, K);
  hpc::copy(s.G, G);
  auto const error_K = std::abs(K[num_points_new - 1] / K0 - 1.0);
  auto const error_G = std::abs(G[num_points_new - 1] / G0 - 1.0);
  auto const eps = hpc::machine_epsilon<double>();
  ASSERT_LE(error_K, eps);
  ASSERT_LE(error_G, eps);
}
