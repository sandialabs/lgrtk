#pragma once

#include <hpc_array_vector.hpp>
#include <hpc_dimensional.hpp>
#include <hpc_vector.hpp>
#include <hpc_vector3.hpp>
#include <lgr_mesh_indices.hpp>

namespace lgr {

struct maxent_interpolator
{
  maxent_interpolator(
      double const gamma_ = 1.5,
      double const h_ = 1.0,
      double const eps_ = 8192 * hpc::machine_epsilon<double>()) :
  gamma(gamma_), h(h_), eps(eps_) {}

  double gamma{1.5};
  double h{1.0};
  double eps{8192 * hpc::machine_epsilon<double>()};

  template <typename Index>
  void operator()(hpc::device_array_vector<hpc::position<double>, Index> const & sources,
      hpc::position<double> const & target,
      hpc::device_vector<hpc::basis_value<double>, Index> & N) const
  {
    auto const beta = gamma / h / h;
    auto converged = false;
    hpc::basis_gradient<double> mu(0.0, 0.0, 0.0);
    using jacobian = hpc::matrix3x3<hpc::quantity<double, hpc::area_dimension>>;
    auto J = jacobian::zero();
    auto iter = 0;
    auto const max_iter = 16;
    while (converged == false) {
      HPC_ASSERT(iter < max_iter, "Exceeded maximum iterations");
      hpc::position<double> R(0.0, 0.0, 0.0);
      auto dRdmu = jacobian::zero();
      for (auto && source : sources) {
        auto const r = source.load() - target;
        auto const rr = hpc::inner_product(r, r);
        auto const mur = hpc::inner_product(mu, r);
        auto const boltzmann_factor = std::exp(-mur - beta * rr);
        R += r * boltzmann_factor;
        dRdmu -= boltzmann_factor * hpc::outer_product(r, r);
      }
      auto const dmu = -hpc::solve_full_pivot(dRdmu, R);
      mu += dmu;
      auto const nmu = hpc::norm(mu);
      auto const ndmu = hpc::norm(dmu);
      auto const error = nmu > hpc::machine_epsilon<double>() ? ndmu / nmu : ndmu;
      converged = (error <= eps);
      J = dRdmu;
      ++iter;
    }
    N.resize(sources.size());
    auto Z = 0.0;
    auto i = 0;
    for (auto && source : sources) {
      auto const r = source.load() - target;
      auto const rr = hpc::inner_product(r, r);
      auto const mur = hpc::inner_product(mu, r);
      auto const boltzmann_factor = std::exp(-mur - beta * rr);
      Z += boltzmann_factor;
      N[i] = boltzmann_factor;
      ++i;
    }
    for (auto & NZ : N) {
      NZ /= Z;
    }
  }
};

} // namespace lgr

