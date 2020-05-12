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

TEST(map, maxenx_interpolation)
{
  lgr::state s;
  hexahedron_eight_points(s);
  lgr::otm_host_pinned_state host_s;
  auto const num_nodes_old = s.nodes.size();
  auto const num_nodes_new = num_nodes_old + 1;
  host_s.x.resize(num_nodes_new);
  hpc::copy(s.x, host_s.x);
  s.x.resize(num_nodes_new);
  s.u.resize(num_nodes_new);
  host_s.x[num_nodes_new - 1] = hpc::position<double>(0,  0,  0);
  hpc::copy(host_s.x, s.x);
  hpc::copy(host_s.x, s.u);
  otm_populate_new_nodes(s, 0, num_nodes_old, num_nodes_old, num_nodes_new);
  hpc::copy(s.u, host_s.x);
  auto const new_u = host_s.x[num_nodes_new - 1].load();
  auto const error = hpc::norm(new_u);
  auto const eps = hpc::machine_epsilon<double>();
  ASSERT_LE(error, eps);
}
