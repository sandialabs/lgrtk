#pragma once

#include <hpc_macros.hpp>
#include <hpc_vector3.hpp>
#include <lgr_domain.hpp>
#include <otm_input.hpp>
#include <lgr_physics.hpp>

namespace lgr {

HPC_NOINLINE inline void const_v(hpc::counting_range<node_index> const nodes,
    hpc::device_array_vector<hpc::position<double>, node_index> const&,  // pos
    hpc::device_array_vector<hpc::velocity<double>, node_index>* v_vector) {
  auto const nodes_to_v = v_vector->begin();
  auto functor = [=] HPC_DEVICE(node_index const node) {
    nodes_to_v[node] = hpc::velocity<double>(100.0, 0.0, 0.0);
  };
  hpc::for_each(hpc::device_policy(), nodes, functor);
}

HPC_NOINLINE inline void otm();
void otm() {
  constexpr material_index body(0);
  constexpr material_index nmaterials(1);
  constexpr material_index x_boundary(1);
  constexpr material_index y_boundary(2);
  constexpr material_index z_boundary(3);
  constexpr material_index nboundaries(3);
  input in(nmaterials, nboundaries);
  in.name = "otm";
  in.end_time = 1.0e-3;
  in.num_file_outputs = 100;
  in.elements_along_x = 1;
  in.x_domain_size = 1.0e-3;
  in.elements_along_y = 1;
  in.y_domain_size = 1.0e-3;
  in.elements_along_z = 1;
  in.z_domain_size = 1.0e-3;
  in.rho0[body] = 1000.0;
  in.enable_neo_Hookean[body] = true;
  in.K0[body] = 1.0e09;
  in.G0[body] = 1.0e09;
  in.initial_v = const_v;
  static constexpr hpc::vector3<double> x_axis(1.0, 0.0, 0.0);
  static constexpr hpc::vector3<double> y_axis(0.0, 1.0, 0.0);
  static constexpr hpc::vector3<double> z_axis(0.0, 0.0, 1.0);
  static constexpr double eps = 1.0e-10;
  auto x_domain = std::make_unique<union_domain>();
  x_domain->add(epsilon_around_plane_domain({x_axis, 0.0}, eps));
  x_domain->add(epsilon_around_plane_domain({x_axis, in.x_domain_size}, eps));
  in.domains[x_boundary] = std::move(x_domain);
  in.zero_acceleration_conditions.push_back({x_boundary, x_axis});
  auto y_domain = std::make_unique<union_domain>();
  y_domain->add(epsilon_around_plane_domain({y_axis, 0.0}, eps));
  y_domain->add(epsilon_around_plane_domain({y_axis, in.y_domain_size}, eps));
  in.domains[y_boundary] = std::move(y_domain);
  in.zero_acceleration_conditions.push_back({y_boundary, y_axis});
  auto z_domain = std::make_unique<union_domain>();
  z_domain->add(epsilon_around_plane_domain({z_axis, 0.0}, eps));
  z_domain->add(epsilon_around_plane_domain({z_axis, in.z_domain_size}, eps));
  in.domains[z_boundary] = std::move(z_domain);
  in.zero_acceleration_conditions.push_back({z_boundary, z_axis});
  run(in);
}

} // namespace lgr
