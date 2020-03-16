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
#include <otm_state.hpp>
#include <unit_tests/otm_unit_mesh.hpp>

TEST(maxent, partition_unity_value_1)
{
  lgr::state s;

  tetrahedron_single_point(s);

  auto num_points = s.points.size();
  double const init = -num_points;
  auto const error = hpc::reduce(hpc::device_policy(), s.N, init) / num_points;
  auto const eps = hpc::machine_epsilon<double>();

  ASSERT_LE(error, eps);
}

TEST(maxent, partition_unity_value_2)
{
  lgr::state s;

  two_tetrahedra_two_points(s);

  auto num_points = s.points.size();
  double const init = -num_points;
  auto const error = hpc::reduce(hpc::device_policy(), s.N, init) / num_points;
  auto const eps = hpc::machine_epsilon<double>();

  ASSERT_LE(error, eps);
}

TEST(maxent, partition_unity_value_3)
{
  lgr::state s;

  hexahedron_eight_points(s);

  auto num_points = s.points.size();
  double const init = -num_points;
  auto const error = hpc::reduce(hpc::device_policy(), s.N, init) / num_points;
  auto const eps = 3 * hpc::machine_epsilon<double>();

  ASSERT_LE(error, eps);
}

namespace lgr_unit {

double compute_basis_gradient_error(const lgr::state& s) {
  auto num_points = s.points.size();
  auto const nodes_to_grad_N = s.grad_N.begin();
  auto const supports = s.points_to_point_nodes.cbegin();
  auto functor = [=] HPC_DEVICE (lgr::point_index const point) {
    auto const support = supports[point];
    hpc::basis_gradient<double> s(0, 0, 0);
    for (auto point_node : support) {
      auto const grad_N = nodes_to_grad_N[point_node].load();
      s += grad_N;
    }
    return hpc::abs(s);
  };
  hpc::basis_gradient<double> init(0, 0, 0);
  auto const errors = hpc::transform_reduce(hpc::device_policy(), s.points, init, hpc::plus<hpc::basis_gradient<double> >(), functor);
  return hpc::norm(errors / num_points) / 3.;
}

}

TEST(maxent, partition_unity_gradient_1)
{
  lgr::state s;

  tetrahedron_single_point(s);

  auto const error = lgr_unit::compute_basis_gradient_error(s);
  auto const eps = hpc::machine_epsilon<double>();

  ASSERT_LE(error, eps);
}

TEST(maxent, partition_unity_gradient_2)
{
  lgr::state s;

  two_tetrahedra_two_points(s);

  auto const error = lgr_unit::compute_basis_gradient_error(s);
  auto const eps = 4 * hpc::machine_epsilon<double>();

  ASSERT_LE(error, eps);
}

TEST(maxent, partition_unity_gradient_3)
{
  lgr::state s;

  hexahedron_eight_points(s);

  auto const error = lgr_unit::compute_basis_gradient_error(s);
  auto const eps = hpc::machine_epsilon<double>();

  ASSERT_LE(error, eps);
}

namespace lgr_unit {

double compute_linear_reproducibility_error(const lgr::state& s) {
  auto num_points = s.points.size();
  auto const points_to_xp = s.xp.begin();
  auto const point_nodes_to_N = s.N.begin();
  auto const nodes_to_x = s.x.begin();
  auto const supports = s.points_to_point_nodes.cbegin();
  auto const points_to_point_nodes = s.point_nodes_to_nodes.cbegin();
  auto functor = [=] HPC_DEVICE (lgr::point_index const point) {
    auto const support = supports[point];
    auto const xp = points_to_xp[point].load();
    hpc::position<double> x(0, 0, 0);
    for (auto point_node : support) {
      auto const node = points_to_point_nodes[point_node];
      auto const xn = nodes_to_x[node].load();
      auto const N = point_nodes_to_N[point_node];
      x += (N * xn);
    }
    return hpc::abs(x - xp);
  };
  hpc::position<double> init(0, 0, 0);
  auto const errors = hpc::transform_reduce(hpc::device_policy(), s.points, init, hpc::plus<hpc::position<double> >(), functor);
  return hpc::norm(errors / (3 * num_points));
}

}

TEST(maxent, linear_reproducibility_1)
{
  lgr::state s;

  tetrahedron_single_point(s);

  auto const error = lgr_unit::compute_linear_reproducibility_error(s);
  auto const eps = hpc::machine_epsilon<double>();

  ASSERT_LE(error, eps);
}

TEST(maxent, linear_reproducibility_2)
{
  lgr::state s;

  two_tetrahedra_two_points(s);

  auto const error = lgr_unit::compute_linear_reproducibility_error(s);
  auto const eps = hpc::machine_epsilon<double>();

  ASSERT_LE(error, eps);
}

TEST(maxent, linear_reproducibility_3)
{
  lgr::state s;

  hexahedron_eight_points(s);

  auto const error = lgr_unit::compute_linear_reproducibility_error(s);
  auto const eps = hpc::machine_epsilon<double>();

  ASSERT_LE(error, eps);
}
