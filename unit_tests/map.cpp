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

template <typename T, typename I>
void
resize_preserve(hpc::device_array_vector<T, I>& v, I const new_size)
{
  auto const old_size = v.size();
  if (old_size == new_size) return;
  hpc::pinned_array_vector<T, I> host_old(old_size);
  hpc::copy(v, host_old);
  v.resize(new_size);
  hpc::pinned_array_vector<T, I> host_new(new_size);
  for (auto i = 0; i < std::min(old_size, new_size); ++i) {
    host_new[i] = host_old[i].load();
  }
  for (auto i = old_size; i < new_size; ++i) { host_new[i] = T::zero(); }
  hpc::copy(host_new, v);
}

template <typename T, typename I>
void
resize_preserve(hpc::device_vector<T, I>& v, I const new_size)
{
  auto const old_size = v.size();
  if (old_size == new_size) return;
  hpc::pinned_vector<T, I> host_old(old_size);
  hpc::copy(v, host_old);
  v.resize(new_size);
  hpc::pinned_vector<T, I> host_new(new_size);
  for (auto i = 0; i < std::min(old_size, new_size); ++i) {
    host_new[i] = host_old[i];
  }
  for (auto i = old_size; i < new_size; ++i) { host_new[i] = T(0.0); }
  hpc::copy(host_new, v);
}

TEST(map, maxent_populate_nodes)
{
  lgr::state s;
  hexahedron_eight_points(s);
  s.otm_beta               = 1.5;
  auto const num_nodes_old = s.nodes.size();
  auto const num_nodes_new = num_nodes_old + 1;
  auto const u0            = hpc::position<double>(12.0e-06, 4.0e-06, 3.0e-06);
  auto const v0            = hpc::velocity<double>(3.0, 4.0, 12.0);
  s.u.resize(num_nodes_old);
  s.v.resize(num_nodes_old);
  hpc::fill(hpc::device_policy(), s.u, u0);
  hpc::fill(hpc::device_policy(), s.v, v0);
  resize_preserve(s.x, num_nodes_new);
  resize_preserve(s.u, num_nodes_new);
  resize_preserve(s.v, num_nodes_new);
  lgr::otm_populate_new_nodes(
      s, 0, num_nodes_old, num_nodes_old, num_nodes_new);
  hpc::pinned_array_vector<hpc::position<double>, lgr::node_index> u(
      num_nodes_new);
  hpc::pinned_array_vector<hpc::position<double>, lgr::node_index> v(
      num_nodes_new);
  hpc::copy(s.u, u);
  hpc::copy(s.v, v);
  auto const error_u =
      std::abs(hpc::norm(u[num_nodes_new - 1].load()) / 13.0e-06 - 1.0);
  auto const error_v =
      std::abs(hpc::norm(v[num_nodes_new - 1].load()) / 13.0 - 1.0);
  auto const eps = hpc::machine_epsilon<double>();
  ASSERT_LE(error_u, eps);
  ASSERT_LE(error_v, eps);
}

TEST(map, maxent_populate_points)
{
  lgr::state s;
  hexahedron_eight_points(s);
  s.otm_beta                = 1.5;
  auto const num_points_old = s.points.size();
  auto const num_points_new = num_points_old + 1;
  auto const K0             = hpc::pressure<double>(2.0e+09);
  auto const G0             = hpc::pressure<double>(1.0e+09);
  auto const rho0           = hpc::density<double>(1.0e+03);
  auto const ep0            = hpc::strain<double>(1.0e-01);
  auto const ep_dot0        = hpc::strain_rate<double>(1.0e+01);
  auto const b0             = hpc::acceleration<double>(0.0, 0.0, -9.81);
  auto const V0 =
      hpc::reduce(hpc::device_policy(), s.V, hpc::volume<double>(0.0));
  auto const r0 = hpc::vector3<double>(0.4, 0.5, 0.6);
  auto const u0 =
      hpc::matrix3x3<double>(1.5, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.5);
  auto const R0  = hpc::rotation_tensor_from_rotation_vector(r0);
  auto const U0  = hpc::exp(u0);
  auto const F0  = R0 * U0;
  auto const rp0 = hpc::vector3<double>(0.1, 0.2, 0.3);
  auto const up0 =
      hpc::matrix3x3<double>(-0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1);
  auto const Rp0 = hpc::rotation_tensor_from_rotation_vector(rp0);
  auto const Up0 = hpc::exp(up0);
  auto const Fp0 = Rp0 * Up0;
  s.K.resize(num_points_old);
  s.G.resize(num_points_old);
  s.rho.resize(num_points_old);
  s.ep.resize(num_points_old);
  s.ep_dot.resize(num_points_old);
  s.b.resize(num_points_old);
  s.V.resize(num_points_old);
  s.F_total.resize(num_points_old);
  s.Fp_total.resize(num_points_old);
  hpc::fill(hpc::device_policy(), s.K, K0);
  hpc::fill(hpc::device_policy(), s.G, G0);
  hpc::fill(hpc::device_policy(), s.rho, rho0);
  hpc::fill(hpc::device_policy(), s.ep, ep0);
  hpc::fill(hpc::device_policy(), s.ep_dot, ep_dot0);
  hpc::fill(hpc::device_policy(), s.b, b0);
  hpc::fill(hpc::device_policy(), s.F_total, F0);
  hpc::fill(hpc::device_policy(), s.Fp_total, Fp0);
  resize_preserve(s.xp, num_points_new);
  resize_preserve(s.K, num_points_new);
  resize_preserve(s.G, num_points_new);
  resize_preserve(s.rho, num_points_new);
  resize_preserve(s.ep, num_points_new);
  resize_preserve(s.ep_dot, num_points_new);
  resize_preserve(s.b, num_points_new);
  resize_preserve(s.V, num_points_new);
  resize_preserve(s.F_total, num_points_new);
  resize_preserve(s.Fp_total, num_points_new);
  lgr::otm_populate_new_points(
      s, 0, num_points_old, num_points_old, num_points_new);
  hpc::pinned_array_vector<hpc::position<double>, lgr::point_index> xp(
      num_points_new);
  hpc::pinned_vector<hpc::pressure<double>, lgr::point_index> K(num_points_new);
  hpc::pinned_vector<hpc::pressure<double>, lgr::point_index> G(num_points_new);
  hpc::pinned_vector<hpc::density<double>, lgr::point_index>  rho(
      num_points_new);
  hpc::pinned_vector<hpc::strain<double>, lgr::point_index> ep(num_points_new);
  hpc::pinned_vector<hpc::strain_rate<double>, lgr::point_index> ep_dot(
      num_points_new);
  hpc::pinned_array_vector<hpc::acceleration<double>, lgr::point_index> b(
      num_points_new);
  hpc::pinned_vector<hpc::volume<double>, lgr::point_index> V(num_points_new);
  hpc::pinned_array_vector<hpc::deformation_gradient<double>, lgr::point_index>
      F(num_points_new);
  hpc::pinned_array_vector<hpc::deformation_gradient<double>, lgr::point_index>
      Fp(num_points_new);
  hpc::copy(s.K, K);
  hpc::copy(s.G, G);
  hpc::copy(s.rho, rho);
  hpc::copy(s.ep, ep);
  hpc::copy(s.ep_dot, ep_dot);
  hpc::copy(s.b, b);
  hpc::copy(s.V, V);
  hpc::copy(s.F_total, F);
  hpc::copy(s.Fp_total, Fp);
  auto const error_K   = std::abs(K[num_points_new - 1] / K0 - 1.0);
  auto const error_G   = std::abs(G[num_points_new - 1] / G0 - 1.0);
  auto const error_rho = std::abs(rho[num_points_new - 1] / rho0 - 1.0);
  auto const error_ep  = std::abs(ep[num_points_new - 1] / ep0 - 1.0);
  auto const error_ep_dot =
      std::abs(ep_dot[num_points_new - 1] / ep_dot0 - 1.0);
  auto const error_b =
      hpc::norm(b[num_points_new - 1].load() - b0) / hpc::norm(b0);
  auto const error_V =
      std::abs(num_points_new * V[num_points_new - 1] / V0 - 1.0);
  auto const error_F =
      hpc::norm(F[num_points_new - 1].load() - F0) / hpc::norm(F0);
  auto const error_Fp =
      hpc::norm(Fp[num_points_new - 1].load() - Fp0) / hpc::norm(Fp0);
  auto const eps = hpc::machine_epsilon<double>();
  ASSERT_LE(error_K, eps);
  ASSERT_LE(error_G, eps);
  ASSERT_LE(error_rho, eps);
  ASSERT_LE(error_ep, eps);
  ASSERT_LE(error_ep_dot, eps);
  ASSERT_LE(error_b, eps);
  ASSERT_LE(error_V, eps);
  ASSERT_LE(error_F, 4 * eps);
  ASSERT_LE(error_Fp, 2 * eps);
}

TEST(map, align_rotation_vectors)
{
  auto const pi          = std::acos(-1.0);
  auto const eps         = hpc::machine_epsilon<double>();
  auto const delta       = 8192 * eps;
  auto const num_vectors = 2;
  hpc::pinned_array_vector<hpc::vector3<double>, int> host_rvs(num_vectors);
  host_rvs[0] = hpc::vector3<double>(pi - delta, 0.0, 0.0);
  host_rvs[1] = hpc::vector3<double>(delta - pi, 0.0, 0.0);
  hpc::device_array_vector<hpc::vector3<double>, int> device_rvs(num_vectors);
  hpc::copy(host_rvs, device_rvs);
  lgr::align_rotation_vectors(device_rvs);
  hpc::copy(device_rvs, host_rvs);
  auto const error =
      std::abs(hpc::norm(host_rvs[1].load()) / (pi + delta) - 1.0);
  ASSERT_LE(error, eps);
}

TEST(map, polar_lie_decompose)
{
  auto const                            range_begin = lgr::point_index(0);
  auto const                            range_end   = lgr::point_index(2);
  hpc::counting_range<lgr::point_index> range(range_begin, range_end);
  hpc::pinned_array_vector<hpc::matrix3x3<double>, lgr::point_index> host_F(
      range_end);
  hpc::device_array_vector<hpc::matrix3x3<double>, lgr::point_index> F(
      range_end);
  hpc::device_array_vector<hpc::vector3<double>, lgr::point_index> r(range_end);
  hpc::device_array_vector<hpc::matrix3x3<double>, lgr::point_index> u(
      range_end);
  auto const r1 = hpc::vector3<double>(0.1, 0.2, 0.3);
  auto const r2 = hpc::vector3<double>(0.4, 0.5, 0.6);
  auto const u1 =
      hpc::matrix3x3<double>(-0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1);
  auto const u2 =
      hpc::matrix3x3<double>(1.5, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.5);
  auto const R1 = hpc::rotation_tensor_from_rotation_vector(r1);
  auto const R2 = hpc::rotation_tensor_from_rotation_vector(r2);
  auto const U1 = hpc::exp(u1);
  auto const U2 = hpc::exp(u2);
  auto const F1 = R1 * U1;
  auto const F2 = R2 * U2;
  host_F[0]     = F1;
  host_F[1]     = F2;
  hpc::copy(host_F, F);
  lgr::polar_lie_decompose(F, r, u, range);
  hpc::pinned_array_vector<hpc::vector3<double>, lgr::point_index> host_r(
      range_end);
  hpc::pinned_array_vector<hpc::matrix3x3<double>, lgr::point_index> host_u(
      range_end);
  hpc::copy(r, host_r);
  hpc::copy(u, host_u);
  auto const eps      = 2 * hpc::machine_epsilon<double>();
  auto const error_r1 = hpc::norm(host_r[0].load() - r1) / hpc::norm(r1);
  auto const error_r2 = hpc::norm(host_r[1].load() - r2) / hpc::norm(r2);
  auto const error_u1 = hpc::norm(host_u[0].load() - u1) / hpc::norm(u1);
  auto const error_u2 = hpc::norm(host_u[1].load() - u2) / hpc::norm(u2);
  ASSERT_LE(error_r1, 2 * eps);
  ASSERT_LE(error_r2, eps);
  ASSERT_LE(error_u1, 15 * eps);
  ASSERT_LE(error_u2, 5 * eps);
}
