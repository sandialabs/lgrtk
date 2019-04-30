#include <lgr_state.hpp>
#include <lgr_input.hpp>

namespace lgr {

void resize_state(input const& in, state& s) {
  s.u.resize(s.nodes.size());
  s.v.resize(s.nodes.size());
  s.V.resize(s.points.size());
  s.grad_N.resize(s.points.size() * s.nodes_in_element.size());
  s.F_total.resize(s.points.size());
  s.sigma.resize(s.points.size());
  s.symm_grad_v.resize(s.points.size());
  if (!in.enable_nodal_energy) s.p.resize(s.points.size());
  s.K.resize(s.points.size());
  s.G.resize(s.points.size());
  s.c.resize(s.points.size());
  s.element_f.resize(s.points.size() * s.nodes_in_element.size());
  s.f.resize(s.nodes.size());
  s.rho.resize(s.points.size());
  if (!in.enable_nodal_energy) s.e.resize(s.points.size());
  s.rho_e_dot.resize(s.points.size());
  s.material_mass.resize(in.materials.size(), s.nodes.size(), s.devpool);
  s.mass.resize(s.nodes.size());
  s.a.resize(s.nodes.size());
  s.h_min.resize(s.elements.size());
  if (in.enable_viscosity) {
    s.h_art.resize(s.elements.size());
  }
  s.nu_art.resize(s.points.size());
  s.element_dt.resize(s.points.size());
  if (in.enable_nodal_pressure) {
    s.p_h.resize(s.nodes.size());
    s.p_h_dot.resize(s.nodes.size());
    s.v_prime.resize(s.points.size());
    s.W.resize(s.points.size() * s.nodes_in_element.size());
  }
  if (in.enable_nodal_energy) {
    s.K_h.resize(s.nodes.size());
    s.p_h.resize(s.nodes.size());
    s.e_h.resize(in.materials.size(), s.nodes.size(), s.devpool);
    s.e_h_dot.resize(s.nodes.size());
    s.rho_h.resize(s.nodes.size());
    s.q.resize(s.points.size());
    s.W.resize(s.points.size() * s.nodes_in_element.size());
  }
  s.material.resize(s.elements.size());
  if (in.enable_adapt) {
    s.quality.resize(s.elements.size());
    s.h_adapt.resize(s.nodes.size());
  }
}

}
