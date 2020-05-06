#pragma once

#include <hpc_array_vector.hpp>
#include <hpc_dimensional.hpp>
#include <hpc_vector.hpp>
#include <hpc_vector3.hpp>
#include <lgr_mesh_indices.hpp>

namespace lgr {

struct maxent_interpolator
{
  maxent_interpolator(double const gamma_, double const h_, double const eps_) :
  gamma(gamma_), h(h_), eps(eps_) {}

  double gamma{1.0};
  double h{1.0};
  double eps{8192 * hpc::machine_epsilon<double>()};

  void operator()(hpc::position<double> const & target,
      hpc::device_array_vector<hpc::position<double>, node_point_index> const &sources,
      hpc::device_vector<hpc::basis_value<double>, node_point_index> &N) const
  {
    auto const beta = gamma / h / h;
    auto converged = false;
    hpc::basis_gradient<double> mu(0.0, 0.0, 0.0);
    using jacobian = hpc::matrix3x3<hpc::quantity<double, hpc::area_dimension>>;
    auto J = jacobian::zero();
    auto iter = 0;
    auto const max_iter = 16;
    while (converged == false) {
      if (iter >= max_iter) HPC_ERROR_EXIT("Exceeded maximum iterations.");
      hpc::position<double> R(0.0, 0.0, 0.0);
      auto dRdmu = jacobian::zero();
      for (auto & source : sources) {
        auto const r = source - target;
        auto const rr = hpc::inner_product(r, r);
        auto const mur = hpc::inner_product(mu, r);
        auto const boltzmann_factor = std::exp(-mur - beta * rr);
        R += r * boltzmann_factor;
        dRdmu -= boltzmann_factor * hpc::outer_product(r, r);
      }
      auto const dmu = -hpc::solve_full_pivot(dRdmu, R);
      mu += dmu;
      auto const error = hpc::norm(dmu) / hpc::norm(mu);
      converged = (error <= eps);
      J = dRdmu;
      ++iter;
    }
  }
};

} // namespace lgr

