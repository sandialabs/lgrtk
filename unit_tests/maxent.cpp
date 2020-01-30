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
#include <unit_tests/otm_unit_mesh.hpp>

TEST(maxent, partition_unity_value_1)
{
  lgr::state s;

  tetrahedron_single_point(s);

  auto num_points = s.points.size();
  double const init = -num_points;
  auto const error = hpc::reduce(hpc::device_policy(), s.N, init);
  auto const eps = hpc::machine_epsilon<double>();

  ASSERT_LE(error, eps);
}

TEST(maxent, partition_unity_value_2)
{
  lgr::state s;

  two_tetrahedra_two_points(s);

  auto num_points = s.points.size();
  double const init = -num_points;
  auto const error = hpc::reduce(hpc::device_policy(), s.N, init);
  auto const eps = hpc::machine_epsilon<double>();

  ASSERT_LE(error, eps);
}

TEST(maxent, partition_unity_gradient_1)
{
  lgr::state s;

  tetrahedron_single_point(s);

  auto num_points = s.points.size();
  auto const point_nodes_to_grad_N = s.grad_N.begin();
  auto const supports = s.points * s.nodes_in_support;
  auto const num_nodes_in_support = s.nodes_in_support.size();
  hpc::basis_gradient<double> errors(0, 0, 0);
  auto functor = [=, &errors] HPC_DEVICE (lgr::point_index const point) {
    auto const support = supports[point];
    hpc::basis_gradient<double> s(0, 0, 0);
    for (auto i = 0; i < num_nodes_in_support; ++i) {
      auto const point_node_index = support[lgr::node_in_support_index(i)];
      auto const grad_N = point_nodes_to_grad_N[point_node_index].load();
      s += grad_N;
    }
    errors += hpc::abs(s);
  };
  hpc::for_each(hpc::device_policy(), s.points, functor);
  errors /= (3 * num_points);
  auto const error = hpc::norm(errors);
  auto const eps = hpc::machine_epsilon<double>();

  ASSERT_LE(error, eps);
}

TEST(maxent, partition_unity_gradient_2)
{
  lgr::state s;

  two_tetrahedra_two_points(s);

  auto num_points = s.points.size();
  auto const point_nodes_to_grad_N = s.grad_N.begin();
  auto const supports = s.points * s.nodes_in_support;
  auto const num_nodes_in_support = s.nodes_in_support.size();
  hpc::basis_gradient<double> errors(0, 0, 0);
  auto functor = [=, &errors] HPC_DEVICE (lgr::point_index const point) {
    auto const support = supports[point];
    hpc::basis_gradient<double> s(0, 0, 0);
    for (auto i = 0; i < num_nodes_in_support; ++i) {
      auto const point_node_index = support[lgr::node_in_support_index(i)];
      auto const grad_N = point_nodes_to_grad_N[point_node_index].load();
      s += grad_N;
    }
    errors += hpc::abs(s);
  };
  hpc::for_each(hpc::device_policy(), s.points, functor);
  errors /= (3 * num_points);
  auto const error = hpc::norm(errors);
  auto const eps = hpc::machine_epsilon<double>();

  ASSERT_LE(error, eps);
}

TEST(maxent, linear_reproducibility_1)
{
  lgr::state s;

  tetrahedron_single_point(s);

  auto num_points = s.points.size();
  auto const points_to_xm = s.xm.begin();
  auto const point_nodes_to_N = s.N.begin();
  auto const nodes_to_x = s.x.begin();
  auto const supports = s.points * s.nodes_in_support;
  auto const num_nodes_in_support = s.nodes_in_support.size();
  auto const support_nodes_to_nodes = s.supports_to_nodes.begin();
  hpc::position<double> errors(0, 0, 0);
  auto functor = [=, &errors] HPC_DEVICE (lgr::point_index const point) {
    auto const support = supports[point];
    auto const xm = points_to_xm[point].load();
    hpc::position<double> x(0, 0, 0);
    for (auto i = 0; i < num_nodes_in_support; ++i) {
      auto const point_node_index = support[lgr::node_in_support_index(i)];
      auto const node = support_nodes_to_nodes[point_node_index];
      auto const xn = nodes_to_x[node].load();
      auto const N = point_nodes_to_N[point_node_index];
      x += (N * xn);
    }
    errors += hpc::abs(x - xm);
  };
  hpc::for_each(hpc::device_policy(), s.points, functor);
  errors /= (3 * num_points);
  auto const error = hpc::norm(errors);
  auto const eps = hpc::machine_epsilon<double>();

  ASSERT_LE(error, eps);
}

TEST(maxent, linear_reproducibility_2)
{
  lgr::state s;

  two_tetrahedra_two_points(s);

  auto num_points = s.points.size();
  auto const points_to_xm = s.xm.begin();
  auto const point_nodes_to_N = s.N.begin();
  auto const nodes_to_x = s.x.begin();
  auto const supports = s.points * s.nodes_in_support;
  auto const num_nodes_in_support = s.nodes_in_support.size();
  auto const support_nodes_to_nodes = s.supports_to_nodes.begin();
  hpc::position<double> errors(0, 0, 0);
#if 0
  std::cout << "*** s.points.size()            : " << s.points.size() << '\n';
  std::cout << "*** s.xm.size()                : " << s.xm.size() << '\n';
  std::cout << "*** s.N.size()                 : " << s.N.size() << '\n';
  std::cout << "*** s.x.size()                 : " << s.x.size() << '\n';
  std::cout << "*** s.nodes_in_support.size()  : " << s.nodes_in_support.size() << '\n';
  std::cout << "*** s.supports_to_nodes.size() : " << s.supports_to_nodes.size() << '\n';
#endif
  auto functor = [=, &errors] HPC_DEVICE (lgr::point_index const point) {
    auto const support = supports[point];
    auto const xm = points_to_xm[point].load();
    hpc::position<double> x(0, 0, 0);
#if 0
    std::cout << '\n';
    std::cout << "*** point : " << point << '\n';
    std::cout << "*** xm    : " << xm(0) << ',' << xm(1) << ',' << xm(2) << '\n';
#endif
    for (auto i = 0; i < num_nodes_in_support; ++i) {
      auto const point_node_index = support[lgr::node_in_support_index(i)];
      auto const node = support_nodes_to_nodes[point_node_index];
      auto const xn = nodes_to_x[node].load();
      auto const N = point_nodes_to_N[point_node_index];
      x += (N * xn);
#if 0
      std::cout << '\n';
      std::cout << "*** i                : " << i << '\n';
      std::cout << "*** NSI(i)           : " << lgr::node_in_support_index(i) << '\n';
      std::cout << "*** point_node_index : " << point_node_index << '\n';
      std::cout << "*** node             : " << node << '\n';
      std::cout << "*** N                : " << N << '\n';
      std::cout << "*** xn               : " << xn(0) << ',' << xn(1) << ',' << xn(2) << '\n';
#endif
    }
    errors += hpc::abs(x - xm);
#if 0
    std::cout << '\n';
    std::cout << "*** x     : " << x(0) << ',' << x(1) << ',' << x(2) << '\n';
#endif
  };
  hpc::for_each(hpc::device_policy(), s.points, functor);
  errors /= (3 * num_points);
  auto const error = hpc::norm(errors);
  auto const eps = hpc::machine_epsilon<double>();

  ASSERT_LE(error, eps);
}
