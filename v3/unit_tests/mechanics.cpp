#include <gtest/gtest.h>
#include <hpc_algorithm.hpp>
#include <hpc_array_traits.hpp>
#include <hpc_array_vector.hpp>
#include <hpc_dimensional.hpp>
#include <hpc_execution.hpp>
#include <hpc_functional.hpp>
#include <hpc_macros.hpp>
#include <hpc_matrix3x3.hpp>
#include <hpc_numeric.hpp>
#include <hpc_range.hpp>
#include <hpc_transform_reduce.hpp>
#include <hpc_vector.hpp>
#include <lgr_input.hpp>
#include <lgr_mesh_indices.hpp>
#include <lgr_state.hpp>
#include <otm_meshless.hpp>
#include <unit_tests/otm_unit_mesh.hpp>
#include <hpc_math.hpp>

namespace lgr_unit {

double compute_total_mass(const lgr::state &s)
{
  auto const rho = s.rho.cbegin();
  auto const V = s.V.cbegin();
  auto mass_func = [=] HPC_DEVICE (lgr::point_index p)
  {
    return rho[p] * V[p];
  };
  double init_mass = 0.0;
  return hpc::transform_reduce(hpc::device_policy(), s.points, init_mass, hpc::plus<double>(),
      mass_func);
}

}

TEST(mechanics, lumped_mass_1)
{
  lgr::state s;

  tetrahedron_single_point(s);
  lgr::otm_update_nodal_mass(s);

  auto const mass = hpc::reduce(hpc::device_policy(), s.mass, 0.0);
  auto const expected_mass = lgr_unit::compute_total_mass(s);
  auto const error = std::abs(mass / expected_mass - 1.0);
  auto const eps = hpc::machine_epsilon<double>();

  ASSERT_LE(error, eps);
}

TEST(mechanics, lumped_mass_2)
{
  lgr::state s;

  two_tetrahedra_two_points(s);
  lgr::otm_update_nodal_mass(s);

  auto const mass = hpc::reduce(hpc::device_policy(), s.mass, 0.0);
  auto const expected_mass = lgr_unit::compute_total_mass(s);
  auto const error = std::abs(mass / expected_mass - 1.0);
  auto const eps = hpc::machine_epsilon<double>();

  ASSERT_LE(error, eps);
}

TEST(mechanics, lumped_mass_3)
{
  lgr::state s;

  hexahedron_eight_points(s);
  lgr::otm_update_nodal_mass(s);

  auto const mass = hpc::reduce(hpc::device_policy(), s.mass, 0.0);
  auto const expected_mass = lgr_unit::compute_total_mass(s);
  auto const error = std::abs(mass / expected_mass - 1.0);
  auto const eps = hpc::machine_epsilon<double>();

  ASSERT_LE(error, eps);
}

namespace lgr_unit {

double compute_sigma_error(const lgr::state& s) {
  auto const points_to_sigma = s.sigma_full.cbegin();
  auto get_error = [=] HPC_DEVICE (lgr::point_index const point) {
    auto const s = points_to_sigma[point].load();
    return s(0,0)+s(0,1)+s(0,2)+s(1,0)+s(1,1)+s(1,2)+s(2,0)+s(2,1)+s(2,2);
  };
  auto error = hpc::transform_reduce(hpc::device_policy(), s.points, 0.0, hpc::plus<double>(), get_error);
  error /= (9 * s.points.size());
  return error;
}

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
  lgr::otm_update_nodal_mass(s);

  auto const num_points = s.points.size();
  auto const num_nodes = s.nodes.size();

  s.u.resize(num_nodes);
  s.v.resize(num_nodes);
  s.lm.resize(num_nodes);
  s.f.resize(num_nodes);
  s.nodal_materials.resize(num_nodes);

  s.b.resize(num_points);

  s.F_total.resize(num_points);
  s.sigma_full.resize(num_points);
  s.K.resize(num_points);
  s.G.resize(num_points);
  s.potential_density.resize(num_points);

  auto const I = hpc::deformation_gradient<double>::identity();
  hpc::fill(hpc::device_policy(), s.F_total, I);

  lgr::otm_initialize_displacement(s);
  lgr::otm_update_reference(s);
  lgr::otm_update_material_state(in, s, 0);

  auto const error = lgr_unit::compute_sigma_error(s);
  auto const tol = 1.0e-06;
  ASSERT_LE(error, tol);
}

TEST(mechanics, unixial_tension)
{
  auto const pass = lgr::otm_j2_uniaxial_patch_test();
  ASSERT_EQ(pass, true);
}
