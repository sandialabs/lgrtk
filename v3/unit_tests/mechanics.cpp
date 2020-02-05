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
#include <lgr_state.hpp>
#include <otm_meshless.hpp>
#include <unit_tests/otm_unit_mesh.hpp>

TEST(mechanics, lumped_mass_1)
{
  lgr::state s;

  tetrahedron_single_point(s);
  lgr::lump_nodal_mass(s);

  auto const mass = hpc::reduce(hpc::device_policy(), s.mass, 0.0);
  auto const rho = s.rho.cbegin();
  auto const V = s.V.cbegin();
  auto expected_mass = 0.0;
  for (auto p = 0; p < s.points.size(); ++p) {
    expected_mass += rho[p] * V[p];
#if 0
    std::cout << "rho[" << p << "] : " << rho[p] << '\n';
    std::cout << "  V[" << p << "] : " << V[p] << '\n';
#endif
  }
  auto const error = std::abs(mass / expected_mass - 1.0);
  auto const eps = hpc::machine_epsilon<double>();

#if 0
  std::cout << '\n';
  std::cout << "*** expected mass : " << expected_mass << '\n';
  std::cout << "*** computed mass : " << mass << '\n';
  for (auto i = 0; i < s.mass.size(); ++i) {
    std::cout << "s.mass[" << i << "] : " << s.mass[i] << '\n';
  }
  std::cout << '\n';
#endif

  ASSERT_LE(error, eps);
}

TEST(mechanics, lumped_mass_2)
{
  lgr::state s;

  two_tetrahedra_two_points(s);
  lgr::lump_nodal_mass(s);

  auto const mass = hpc::reduce(hpc::device_policy(), s.mass, 0.0);
  auto const rho = s.rho.cbegin();
  auto const V = s.V.cbegin();
  auto expected_mass = 0.0;
  for (auto p = 0; p < s.points.size(); ++p) {
    expected_mass += rho[p] * V[p];
#if 0
    std::cout << "rho[" << p << "] : " << rho[p] << '\n';
    std::cout << "  V[" << p << "] : " << V[p] << '\n';
#endif
  }
  auto const error = std::abs(mass / expected_mass - 1.0);
  auto const eps = hpc::machine_epsilon<double>();

#if 0
  std::cout << '\n';
  std::cout << "*** expected mass : " << expected_mass << '\n';
  std::cout << "*** computed mass : " << mass << '\n';
  for (auto i = 0; i < s.mass.size(); ++i) {
    std::cout << "s.mass[" << i << "] : " << s.mass[i] << '\n';
  }
  std::cout << '\n';
#endif

  ASSERT_LE(error, eps);
}

TEST(mechanics, lumped_mass_3)
{
  lgr::state s;

  hexahedron_eight_points(s);
  lgr::lump_nodal_mass(s);

  auto const mass = hpc::reduce(hpc::device_policy(), s.mass, 0.0);
  auto const rho = s.rho.cbegin();
  auto const V = s.V.cbegin();
  auto expected_mass = 0.0;
  for (auto p = 0; p < s.points.size(); ++p) {
    expected_mass += rho[p] * V[p];
#if 0
    std::cout << "rho[" << p << "] : " << rho[p] << '\n';
    std::cout << "  V[" << p << "] : " << V[p] << '\n';
#endif
  }
  auto const error = std::abs(mass / expected_mass - 1.0);
  auto const eps = hpc::machine_epsilon<double>();

#if 0
  std::cout << '\n';
  std::cout << "*** expected mass : " << expected_mass << '\n';
  std::cout << "*** computed mass : " << mass << '\n';
  for (auto i = 0; i < s.mass.size(); ++i) {
    std::cout << "s.mass[" << i << "] : " << s.mass[i] << '\n';
  }
  std::cout << '\n';
#endif

  ASSERT_LE(error, eps);
}
