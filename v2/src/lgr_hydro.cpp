#include <lgr_hydro.hpp>
#include <lgr_simulation.hpp>
#include <lgr_scope.hpp>
#include <Omega_h_align.hpp>
#include <lgr_for.hpp>
#include <lgr_element_functions.hpp>

namespace lgr {

template <class Elem>
void initialize_configuration(Simulation& sim) {
  LGR_SCOPE(sim);
  auto const points_to_gradients = sim.set(sim.gradient);
  auto const points_to_weights = sim.set(sim.weight);
  auto const nodes_to_x = sim.get(sim.position);
  auto const elems_to_nodes = sim.elems_to_nodes();
  auto const elems_to_time_len = sim.set(sim.time_step_length);
  auto const elems_to_visc_len = sim.set(sim.viscosity_length);
  auto functor = OMEGA_H_LAMBDA(int const elem) {
    auto const elem_nodes = getnodes<Elem>(elems_to_nodes, elem);
    auto const x = getvecs<Elem>(nodes_to_x, elem_nodes);
    auto const shape = Elem::shape(x);
    elems_to_time_len[elem] = shape.lengths.time_step_length;
    elems_to_visc_len[elem] = shape.lengths.viscosity_length;
    for (int elem_pt = 0; elem_pt < Elem::points; ++elem_pt) {
      auto const pt = elem * Elem::points + elem_pt;
      setgrads<Elem>(points_to_gradients, pt,
          shape.basis_gradients[elem_pt]);
      points_to_weights[pt] = shape.weights[elem_pt];
    }
  };
  parallel_for(sim.elems(), std::move(functor));
}

template <class Elem>
void lump_masses(Simulation& sim) {
  LGR_SCOPE(sim);
  auto const points_to_rho = sim.get(sim.density);
  auto const points_to_w = sim.get(sim.weight);
  auto const nodes_to_elems = sim.nodes_to_elems();
  auto const nodes_to_mass = sim.set(sim.nodal_mass);
  auto functor = OMEGA_H_LAMBDA(int node) {
    double node_mass = 0.0;
    for (auto node_elem = nodes_to_elems.a2ab[node];
        node_elem < nodes_to_elems.a2ab[node + 1];
        ++node_elem) {
      auto const elem = nodes_to_elems.ab2b[node_elem];
      auto const code = nodes_to_elems.codes[node_elem];
      auto const elem_node = Omega_h::code_which_down(code);
      double elem_mass = 0.0;
      for (int elem_pt = 0; elem_pt < Elem::points; ++elem_pt) {
        auto const point = elem * Elem::points + elem_pt;
        auto const rho = points_to_rho[point];
        auto const w = points_to_w[point];
        elem_mass += rho * w;
      }
      node_mass += elem_mass * Elem::lumping_factor(elem_node);
    }
    nodes_to_mass[node] = node_mass;
  };
  parallel_for(sim.nodes(), std::move(functor));
}

template <class Elem>
void update_position(Simulation& sim) {
  LGR_SCOPE(sim);
  auto const nodes_to_v = sim.getset(sim.velocity);
  auto const nodes_to_x = sim.getset(sim.position);
  auto const nodes_to_a = sim.get(sim.acceleration);
  auto const dt = sim.dt;
  auto functor = OMEGA_H_LAMBDA(int const node) {
    auto const v_n = getvec<Elem>(nodes_to_v, node);
    auto const x_n = getvec<Elem>(nodes_to_x, node);
    auto const a_n = getvec<Elem>(nodes_to_a, node);
    auto const v_np12 = v_n + (dt / 2.0) * a_n;
    auto const x_np1 = x_n + dt * v_np12;
    setvec<Elem>(nodes_to_v, node, v_np12);
    setvec<Elem>(nodes_to_x, node, x_np1);
  };
  parallel_for(sim.nodes(), std::move(functor));
}

template <class Elem>
void update_configuration(Simulation& sim) {
  LGR_SCOPE(sim);
  auto const elems_to_nodes = sim.disc.ents_to_nodes(ELEMS);
  auto const nodes_to_x = sim.get(sim.position);
  auto const points_to_gradients = sim.set(sim.gradient);
  auto const points_to_weights = sim.getset(sim.weight);
  auto const points_to_rho = sim.getset(sim.density);
  auto const elems_to_time_len = sim.set(sim.time_step_length);
  auto const elems_to_visc_len = sim.set(sim.viscosity_length);
  auto functor = OMEGA_H_LAMBDA(int const elem) {
    auto const elem_nodes = getnodes<Elem>(elems_to_nodes, elem);
    auto const x = getvecs<Elem>(nodes_to_x, elem_nodes);
    auto const shape = Elem::shape(x);
    elems_to_time_len[elem] = shape.lengths.time_step_length;
    elems_to_visc_len[elem] = shape.lengths.viscosity_length;
    for (int elem_pt = 0; elem_pt < Elem::points; ++elem_pt) {
      auto const pt = elem * Elem::points + elem_pt;
      setgrads<Elem>(points_to_gradients, pt,
          shape.basis_gradients[elem_pt]);
      auto const w_n = points_to_weights[pt];
      auto const rho_n = points_to_rho[pt];
      auto const m = w_n * rho_n;
      auto const w_np1 = shape.weights[elem_pt];
      auto const rho_np1 = m / w_np1;
      points_to_weights[pt] = w_np1;
      points_to_rho[pt] = rho_np1;
    }
  };
  parallel_for(sim.elems(), std::move(functor));
}

template <class Elem>
void correct_velocity(Simulation& sim) {
  LGR_SCOPE(sim);
  auto const nodes_to_v = sim.getset(sim.velocity);
  auto const nodes_to_a = sim.get(sim.acceleration);
  auto const dt = sim.dt;
  auto functor = OMEGA_H_LAMBDA(int const node) {
    auto const v_np12 = getvec<Elem>(nodes_to_v, node);
    auto const a_np1 = getvec<Elem>(nodes_to_a, node);
    auto const v_np1 = v_np12 + (dt / 2.0) * a_np1;
    setvec<Elem>(nodes_to_v, node, v_np1);
  };
  parallel_for(sim.nodes(), std::move(functor));
}

template <class Elem>
void compute_stress_divergence(Simulation& sim) {
  LGR_SCOPE(sim);
  auto const points_to_sigma = sim.get(sim.stress);
  auto const points_to_grads = sim.get(sim.gradient);
  auto const points_to_weights = sim.set(sim.weight);
  auto const nodes_to_f = sim.set(sim.force);
  auto const nodes_to_elems = sim.nodes_to_elems();
  auto functor = OMEGA_H_LAMBDA(int const node) {
    auto node_f = zero_vector<Elem::dim>();
    auto const begin = nodes_to_elems.a2ab[node];
    auto const end = nodes_to_elems.a2ab[node + 1];
    for (auto node_elem = begin; node_elem < end; ++node_elem) {
      auto const elem = nodes_to_elems.ab2b[node_elem];
      auto const code = nodes_to_elems.codes[node_elem];
      auto const elem_node = Omega_h::code_which_down(code);
      for (int elem_pt = 0; elem_pt < Elem::points; ++elem_pt) {
        auto const point = elem * Elem::points + elem_pt;
        auto const grad = getvec<Elem>(points_to_grads, point * Elem::nodes + elem_node);
        auto const sigma = getsymm<Elem>(points_to_sigma, point);
        auto const weight = points_to_weights[point];
        auto const cell_f = - (sigma * grad) * weight;
        node_f += cell_f;
      }
    }
    setvec<Elem>(nodes_to_f, node, node_f);
  };
  parallel_for(sim.nodes(), std::move(functor));
}

template <class Elem>
void compute_nodal_acceleration(Simulation& sim) {
  LGR_SCOPE(sim);
  auto const nodes_to_f = sim.get(sim.force);
  auto const nodes_to_m = sim.get(sim.nodal_mass);
  auto const nodes_to_a = sim.set(sim.acceleration);
  auto functor = OMEGA_H_LAMBDA(int const node) {
    auto const f = getvec<Elem>(nodes_to_f, node);
    auto const m = nodes_to_m[node];
    auto const a = f / m;
    setvec<Elem>(nodes_to_a, node, a);
  };
  parallel_for(sim.nodes(), std::move(functor));
}

void apply_tractions(Simulation& /*sim*/) {
//if (!sim.has(sim.traction)) return;
//apply_conditions(sim, sim.traction);
}

template <class Elem>
void compute_point_time_steps(Simulation& sim) {
  LGR_SCOPE(sim);
  auto const points_to_c = sim.get(sim.wave_speed);
  auto const elems_to_h = sim.get(sim.time_step_length);
  auto const points_to_dt = sim.set(sim.point_time_step);
  double const max = std::numeric_limits<double>::max();
  auto functor = OMEGA_H_LAMBDA(int const point) {
    auto const elem = point / Elem::points;
    auto const h = elems_to_h[elem];
    OMEGA_H_CHECK(h > 0.0);
    auto const c = points_to_c[point];
    OMEGA_H_CHECK(c >= 0.0);
    auto const dt = (c == 0.0) ? max : (h / c);
    OMEGA_H_CHECK(dt > 0.0);
    points_to_dt[point] = dt;
  };
  parallel_for(sim.points(), std::move(functor));
}

#define LGR_EXPL_INST(Elem) \
template void initialize_configuration<Elem>(Simulation& sim); \
template void lump_masses<Elem>(Simulation& sim); \
template void update_position<Elem>(Simulation& sim); \
template void update_configuration<Elem>(Simulation& sim); \
template void correct_velocity<Elem>(Simulation& sim); \
template void compute_stress_divergence<Elem>(Simulation& sim); \
template void compute_nodal_acceleration<Elem>(Simulation& sim); \
template void compute_point_time_steps<Elem>(Simulation& sim);
LGR_EXPL_INST_ELEMS
#undef LGR_EXPL_INST

}
