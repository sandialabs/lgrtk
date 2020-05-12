#pragma once

#include <hpc_array.hpp>
#include <hpc_array_traits.hpp>
#include <hpc_array_vector.hpp>
#include <hpc_dimensional.hpp>
#include <hpc_macros.hpp>
#include <hpc_matrix3x3.hpp>
#include <hpc_quaternion.hpp>
#include <hpc_range.hpp>
#include <hpc_symmetric3x3.hpp>
#include <hpc_vector.hpp>
#include <hpc_vector3.hpp>
#include <lgr_mesh_indices.hpp>
#include <cmath>

namespace lgr {
class state;
}

namespace lgr {

#if 0

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
      for (auto source : sources) {
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

#endif

enum adapt_op {
  NONE,
  SPLIT,
  COLLAPSE,
};

struct otm_adapt_state {
  hpc::device_vector<hpc::length<double>, node_index> node_criteria;
  hpc::device_vector<hpc::length<double>, node_index> point_criteria;
  hpc::device_vector<node_index, node_index> other_node;
  hpc::device_vector<point_index, point_index> other_point;
  hpc::device_vector<adapt_op, node_index> node_op;
  hpc::device_vector<adapt_op, point_index> point_op;
  hpc::device_vector<point_index, point_index> point_counts;
  hpc::device_vector<node_index, node_index> node_counts;
  hpc::device_vector<point_index, point_index> old_points_to_new_points;
  hpc::device_vector<node_index, node_index> old_nodes_to_new_nodes;
  hpc::device_vector<point_index, point_index> new_points_to_old_points;
  hpc::device_vector<node_index, node_index> new_nodes_to_old_nodes;
  hpc::device_vector<node_index, point_node_index> new_point_nodes_to_nodes;
  hpc::device_vector<bool, point_index> new_points_are_same;
  hpc::device_vector<bool, node_index> new_nodes_are_same;
  hpc::device_array_vector<hpc::array<node_index, 2, int>, node_index> interpolate_from_nodes;
  hpc::device_array_vector<hpc::array<point_index, 2, int>, point_index> interpolate_from_points;
  hpc::counting_range<point_index> new_points;
  hpc::counting_range<node_index> new_nodes;

  otm_adapt_state(state const&);
};

} // namespace lgr

