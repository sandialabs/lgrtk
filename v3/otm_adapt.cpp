#include <hpc_array_vector.hpp>
#include <hpc_dimensional.hpp>
#include <hpc_vector.hpp>
#include <hpc_vector3.hpp>
#include <lgr_mesh_indices.hpp>
#include <otm_adapt.hpp>
#include <lgr_state.hpp>

namespace lgr {

void otm_populate_new_nodes(state & s,
    node_index begin_src, node_index end_src,
    node_index begin_target, node_index end_target)
{
  hpc::counting_range<node_index> source_range(begin_src, end_src);
  hpc::counting_range<node_index> target_range(begin_target, end_target);
  hpc::device_vector<hpc::basis_value<double>, node_index> NZ(source_range.size());
  auto const nodes_to_x = s.x.cbegin();
  auto const nodes_to_u = s.u.begin();
  auto const index_to_NZ = NZ.begin();
  auto maxent_interpolator = [=] HPC_DEVICE (node_index const node) {
    auto const target = nodes_to_x[node].load();
    auto const gamma = 1.5;
    auto const h = 1.0;
    auto const eps = 8192 * hpc::machine_epsilon<double>();
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
      for (auto && source_index : source_range) {
        auto const r = nodes_to_x[source_index].load() - target;
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
    auto Z = 0.0;
    auto i = 0;
    for (auto && source_index : source_range) {
      auto const r = nodes_to_x[source_index].load() - target;
      auto const rr = hpc::inner_product(r, r);
      auto const mur = hpc::inner_product(mu, r);
      auto const boltzmann_factor = std::exp(-mur - beta * rr);
      Z += boltzmann_factor;
      index_to_NZ[i] = boltzmann_factor;
      ++i;
    }
    i = 0;
    auto node_u = hpc::displacement<double>::zero();
    for (auto && source_index : source_range) {
      auto const u = nodes_to_x[source_index].load();
      auto const N = index_to_NZ[i] / Z;
      node_u = node_u + N * u;
      ++i;
    }
    nodes_to_u[node] = node_u;
  };
  hpc::for_each(hpc::device_policy(), target_range, maxent_interpolator);
}

} // namespace lgr

