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
  auto const expected_mass = s.rho.cbegin()[0] * s.V.cbegin()[0];
  auto const error = std::abs(mass / expected_mass - 1);
  auto const eps = hpc::machine_epsilon<double>();

#if 0
  std::cout << '\n';
  std::cout << "*** mass : " << mass << '\n';
  for (auto i = 0; i < s.mass.size(); ++i) {
    std::cout << "s.mass[" << i << "] : " << s.mass[i] << '\n';
  }
  std::cout << '\n';
#endif

  ASSERT_LE(error, eps);
}
