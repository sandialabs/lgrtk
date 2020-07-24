#pragma once

#include <hpc_macros.hpp>
#include <lgr_state.hpp>

namespace lgr {

HPC_NOINLINE inline void
update_c(state& s)
{
  auto const points_to_rho = s.rho.cbegin();
  auto const points_to_K   = s.K.cbegin();
  auto const points_to_G   = s.G.cbegin();
  auto const points_to_c   = s.c.begin();
  auto       functor       = [=] HPC_DEVICE(point_index const point) {
    auto const rho     = points_to_rho[point];
    auto const K       = points_to_K[point];
    auto const G       = points_to_G[point];
    auto const M       = K + (4.0 / 3.0) * G;
    auto const c       = sqrt(M / rho);
    points_to_c[point] = c;
  };
  hpc::for_each(hpc::device_policy(), s.points, functor);
}

HPC_NOINLINE inline void
find_max_stable_dt(state& s)
{
  hpc::time<double> const init(std::numeric_limits<double>::max());
  s.max_stable_dt = hpc::transform_reduce(
      hpc::device_policy(), s.element_dt, init, hpc::minimum<hpc::time<double>>(), hpc::identity<hpc::time<double>>());
  assert(s.max_stable_dt < 1.0);
}

}  // namespace lgr
