#include <gtest/gtest.h>
#include <hpc_matrix3x3.hpp>
#include <hpc_transform_reduce.hpp>
#include <hpc_vector3.hpp>
#include <otm_adapt.hpp>
#include <otm_util.hpp>

TEST(map, maxenx_interpolation)
{
  hpc::device_array_vector<hpc::position<double>, lgr::node_index> src_pos;
  auto const num_sources = lgr::node_index(8);
  auto const sources = hpc::make_counting_range(num_sources);
  src_pos.resize(num_sources);
  src_pos[0] = hpc::position<double>(-1, -1, -1);
  src_pos[1] = hpc::position<double>( 1, -1, -1);
  src_pos[2] = hpc::position<double>( 1,  1, -1);
  src_pos[3] = hpc::position<double>(-1,  1, -1);
  src_pos[4] = hpc::position<double>(-1, -1,  1);
  src_pos[5] = hpc::position<double>( 1, -1,  1);
  src_pos[6] = hpc::position<double>( 1,  1,  1);
  src_pos[7] = hpc::position<double>(-1,  1,  1);
  auto const tgt_pos = hpc::position<double>(0,  0,  0);
  hpc::device_vector<hpc::basis_value<double>, lgr::node_index> N;
  lgr::maxent_interpolator interpolator;
  interpolator(src_pos, tgt_pos, N);
  auto const source_to_N = N.cbegin();
  auto functor = [=] HPC_DEVICE (lgr::node_index const src) {
    auto const e = source_to_N[src] - 0.125;
    return e * e;;
  };
  auto const e2 = hpc::transform_reduce(hpc::device_policy(), sources, 0, hpc::plus<int>(), functor);
  auto const error = std::sqrt(e2);
  auto const eps = hpc::machine_epsilon<double>();
  ASSERT_LE(error, eps);
}
