#include <gtest/gtest.h>
#include <hpc_algorithm.hpp>
#include <hpc_array_traits.hpp>
#include <hpc_array_vector.hpp>
#include <hpc_dimensional.hpp>
#include <hpc_execution.hpp>
#include <hpc_macros.hpp>
#include <hpc_numeric.hpp>
#include <hpc_range.hpp>
#include <hpc_vector.hpp>
#include <hpc_vector3.hpp>
#include <lgr_mesh_indices.hpp>
#include <otm_meshless.hpp>
#include <otm_state.hpp>
#include <otm_util.hpp>
#include <unit_tests/otm_unit_mesh.hpp>

TEST(mechanics, lumped_mass_1)
{
  lgr::state s;

  tetrahedron_single_point(s);
  lgr::otm_lump_nodal_mass(s);

  auto const mass = hpc::reduce(hpc::device_policy(), s.mass, 0.0);
  auto const rho = s.rho.cbegin();
  auto const V = s.V.cbegin();
  auto expected_mass = 0.0;
  for (auto p = 0; p < s.points.size(); ++p) {
    expected_mass += rho[p] * V[p];
  }
  auto const error = std::abs(mass / expected_mass - 1.0);
  auto const eps = hpc::machine_epsilon<double>();

  ASSERT_LE(error, eps);
}

TEST(mechanics, lumped_mass_2)
{
  lgr::state s;

  two_tetrahedra_two_points(s);
  lgr::otm_lump_nodal_mass(s);

  auto const mass = hpc::reduce(hpc::device_policy(), s.mass, 0.0);
  auto const rho = s.rho.cbegin();
  auto const V = s.V.cbegin();
  auto expected_mass = 0.0;
  for (auto p = 0; p < s.points.size(); ++p) {
    expected_mass += rho[p] * V[p];
  }
  auto const error = std::abs(mass / expected_mass - 1.0);
  auto const eps = hpc::machine_epsilon<double>();

  ASSERT_LE(error, eps);
}

TEST(mechanics, lumped_mass_3)
{
  lgr::state s;

  hexahedron_eight_points(s);
  lgr::otm_lump_nodal_mass(s);

  auto const mass = hpc::reduce(hpc::device_policy(), s.mass, 0.0);
  auto const rho = s.rho.cbegin();
  auto const V = s.V.cbegin();
  auto expected_mass = 0.0;
  for (auto p = 0; p < s.points.size(); ++p) {
    expected_mass += rho[p] * V[p];
  }
  auto const error = std::abs(mass / expected_mass - 1.0);
  auto const eps = hpc::machine_epsilon<double>();

  ASSERT_LE(error, eps);
}

TEST(mechanics, hex_translation)
{
  lgr::state s;

  lgr::material_index const num_materials = 1;
  lgr::material_index const num_boundaries = 0;

  lgr::input in(num_materials, num_boundaries);

  in.enable_neo_Hookean.resize(num_materials);
  in.enable_variational_J2.resize(num_materials);
  in.K0.resize(num_materials);
  in.G0.resize(num_materials);

  in.enable_neo_Hookean[0] = true;
  in.enable_variational_J2[0] = false;
  in.K0[0] = hpc::pressure<double>(1.0e+09);
  in.G0[0] = hpc::pressure<double>(1.0e+09);

  hexahedron_eight_points(s);
  lgr::otm_lump_nodal_mass(s);

  auto const num_points = s.points.size();
  auto const num_nodes = s.nodes.size();

  s.u.resize(num_nodes);
  s.v.resize(num_nodes);
  s.lm.resize(num_nodes);
  s.f.resize(num_nodes);

  s.b.resize(num_points);

  s.F_total.resize(num_points);
  s.sigma.resize(num_points);
  s.K.resize(num_points);
  s.G.resize(num_points);
  s.potential_density.resize(num_points);

  lgr::otm_initialize_u(s);
  lgr::otm_initialize_F(s);
  lgr::otm_update_reference(s);
  lgr::otm_update_material_state(in, s, 0);

#if 0
  auto const nodes_to_x = s.x.cbegin();
  auto print_x = [=] HPC_HOST (lgr::node_index const node) {
    auto const x = nodes_to_x[node].load();
    HPC_TRACE("node: " << node << ", x:\n" << x);
  };
  hpc::for_each(hpc::device_policy(), s.nodes, print_x);

  auto const points_to_xp = s.xp.cbegin();
  auto print_xp = [=] HPC_HOST (lgr::point_index const point) {
    auto const xp = points_to_xp[point].load();
    HPC_TRACE("point: " << point << ", xp:\n" << xp);
  };
  hpc::for_each(hpc::device_policy(), s.points, print_xp);

  auto const nodes_to_u = s.u.cbegin();
  auto print_u = [=] HPC_HOST (lgr::node_index const node) {
    auto const u = nodes_to_u[node].load();
    HPC_TRACE("node: " << node << ", u:\n" << u);
  };
  hpc::for_each(hpc::device_policy(), s.nodes, print_u);

  auto const points_to_F = s.F_total.cbegin();
  auto print_F = [=] HPC_HOST (lgr::point_index const point) {
    auto const F = points_to_F[point].load();
    HPC_TRACE("point: " << point << ", F:\n" << F);
  };
  hpc::for_each(hpc::device_policy(), s.points, print_F);

  auto const points_to_sigma = s.sigma.cbegin();
  auto const points_to_K = s.K.cbegin();
  auto const points_to_G = s.G.cbegin();
  auto print_sigma = [=] HPC_HOST (lgr::point_index const point) {
    auto const sigma = points_to_sigma[point].load();
    auto const K = points_to_K[point];
    auto const G = points_to_G[point];
    HPC_TRACE("point: " << point << ", K: " << K << ", G: " << G << ", sigma:\n" << sigma);
  };
  hpc::for_each(hpc::device_policy(), s.points, print_sigma);
#endif

  auto error = 0.0;
  auto const points_to_sigma = s.sigma.cbegin();
  auto get_error = [=, &error] HPC_HOST (lgr::point_index const point) {
    auto const s = points_to_sigma[point].load();
    error += s(0,0)+s(0,1)+s(0,2)+s(1,0)+s(1,1)+s(1,2)+s(2,0)+s(2,1)+s(2,2);
  };
  hpc::for_each(hpc::device_policy(), s.points, get_error);
  error /= (9 * num_points);
  auto const tol = 1.0e-06;
  ASSERT_LE(error, tol);
}
