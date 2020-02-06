#include <hpc_algorithm.hpp>
#include <hpc_execution.hpp>
#include <hpc_vector.hpp>
#include <lgr_input.hpp>
#include <lgr_state.hpp>

namespace lgr {

void resize_state(input const& in, state& s) {
  auto const is_meshless = in.element == MESHLESS;
  if (is_meshless == true) {
    resize_state_meshless(in, s);
    return;
  }
  s.u.resize(s.nodes.size());
  s.v.resize(s.nodes.size());
  s.V.resize(s.points.size());
  s.grad_N.resize(s.points.size() * s.nodes_in_element.size());
  s.F_total.resize(s.points.size());
  s.sigma.resize(s.points.size());
  s.symm_grad_v.resize(s.points.size());
  auto have_nodal_pressure_or_energy = [&] (material_index const material) {
    return in.enable_nodal_pressure[material] || in.enable_nodal_energy[material];
  };
  if (!hpc::all_of(hpc::serial_policy(), in.materials, have_nodal_pressure_or_energy)) {
    s.p.resize(s.points.size());
  }
  s.K.resize(s.points.size());
  s.G.resize(s.points.size());
  s.c.resize(s.points.size());
  s.element_f.resize(s.points.size() * s.nodes_in_element.size());
  s.f.resize(s.nodes.size());
  s.rho.resize(s.points.size());
  if (!hpc::all_of(hpc::serial_policy(), in.enable_nodal_energy)) {
    s.e.resize(s.points.size());
  }
  s.rho_e_dot.resize(s.points.size());
  {
    // Hyper EP model
    s.Fp_total.resize(s.points.size());
    s.ep.resize(s.points.size());
    s.ep_dot.resize(s.points.size());
    s.dp.resize(s.points.size());
    s.localized.resize(s.points.size());
  }
  s.material_mass.resize(in.materials.size());
  for (auto& mm : s.material_mass) mm.resize(s.nodes.size());
  s.mass.resize(s.nodes.size());
  s.a.resize(s.nodes.size());
  s.h_min.resize(s.elements.size());
  if (in.enable_viscosity) {
    s.h_art.resize(s.elements.size());
  }
  s.nu_art.resize(s.points.size());
  s.element_dt.resize(s.points.size());
  s.p_h.resize(in.materials.size());
  s.p_h_dot.resize(in.materials.size());
  s.e_h.resize(in.materials.size());
  s.e_h_dot.resize(in.materials.size());
  s.rho_h.resize(in.materials.size());
  s.K_h.resize(in.materials.size());
  s.dp_de_h.resize(in.materials.size());
  s.temp.resize(in.materials.size());
  s.ep_h.resize(in.materials.size());
  for (auto const material : in.materials) {
    if (in.enable_nodal_pressure[material]) {
      s.p_h[material].resize(s.nodes.size());
      s.p_h_dot[material].resize(s.nodes.size());
      s.v_prime.resize(s.points.size());
      s.W.resize(s.points.size() * s.nodes_in_element.size());
    }
    if (in.enable_nodal_energy[material]) {
      s.p_h[material].resize(s.nodes.size());
      s.e_h[material].resize(s.nodes.size());
      s.e_h_dot[material].resize(s.nodes.size());
      s.rho_h[material].resize(s.nodes.size());
      s.K_h[material].resize(s.nodes.size());
      s.q.resize(s.points.size());
      s.W.resize(s.points.size() * s.nodes_in_element.size());
      s.dp_de_h[material].resize(s.nodes.size());
      if (in.enable_p_prime[material]) {
        s.p_prime.resize(s.points.size());
      }
    }
    if (in.enable_hyper_ep[material]) {
      s.ep_h[material].resize(s.nodes.size());
    }
  }
  s.material.resize(s.elements.size());
  if (in.enable_adapt) {
    s.quality.resize(s.elements.size());
    s.h_adapt.resize(s.nodes.size());
  }
}

void resize_state_meshless(input const& in, state& s) {
  s.u.resize(s.nodes.size());
  s.v.resize(s.nodes.size());
  s.V.resize(s.points.size());
  s.grad_N.resize(s.points.size() * s.points_to_point_nodes.size());
  s.N.resize(s.points.size() * s.points_to_point_nodes.size());
  s.F_total.resize(s.points.size());
  s.sigma.resize(s.points.size());
  s.symm_grad_v.resize(s.points.size());
  s.K.resize(s.points.size());
  s.G.resize(s.points.size());
  s.c.resize(s.points.size());
  s.element_f.resize(s.points.size() * s.nodes_in_element.size());
  s.f.resize(s.nodes.size());
  s.rho.resize(s.points.size());
  s.material_mass.resize(in.materials.size());
  for (auto& mm : s.material_mass) mm.resize(s.nodes.size());
  s.mass.resize(s.nodes.size());
  s.a.resize(s.nodes.size());
  s.h_min.resize(s.elements.size());
  if (in.enable_viscosity) {
    s.h_art.resize(s.elements.size());
  }
  s.nu_art.resize(s.points.size());
  s.element_dt.resize(s.points.size());
  s.p_h.resize(in.materials.size());
  s.p_h_dot.resize(in.materials.size());
  s.e_h.resize(in.materials.size());
  s.e_h_dot.resize(in.materials.size());
  s.rho_h.resize(in.materials.size());
  s.K_h.resize(in.materials.size());
  s.dp_de_h.resize(in.materials.size());
  s.temp.resize(in.materials.size());
  s.ep_h.resize(in.materials.size());
  for (auto const material : in.materials) {
    if (in.enable_nodal_pressure[material]) {
      s.p_h[material].resize(s.nodes.size());
      s.p_h_dot[material].resize(s.nodes.size());
      s.v_prime.resize(s.points.size());
      s.W.resize(s.points.size() * s.nodes_in_element.size());
    }
    if (in.enable_nodal_energy[material]) {
      s.p_h[material].resize(s.nodes.size());
      s.e_h[material].resize(s.nodes.size());
      s.e_h_dot[material].resize(s.nodes.size());
      s.rho_h[material].resize(s.nodes.size());
      s.K_h[material].resize(s.nodes.size());
      s.q.resize(s.points.size());
      s.W.resize(s.points.size() * s.nodes_in_element.size());
      s.dp_de_h[material].resize(s.nodes.size());
      if (in.enable_p_prime[material]) {
        s.p_prime.resize(s.points.size());
      }
    }
    if (in.enable_hyper_ep[material]) {
      s.ep_h[material].resize(s.nodes.size());
    }
  }
  s.material.resize(s.elements.size());
  if (in.enable_adapt) {
    s.quality.resize(s.elements.size());
    s.h_adapt.resize(s.nodes.size());
  }
}
}
