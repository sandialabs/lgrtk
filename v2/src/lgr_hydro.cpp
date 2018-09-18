#include <lgr_hydro.hpp>
#include <lgr_simulation.hpp>
#include <lgr_scope.hpp>
#include <Omega_h_align.hpp>
#include <lgr_for.hpp>

namespace lgr {

template <class Elem>
void initialize_configuration(Simulation& sim) {
  LGR_SCOPE(sim);
  auto points_to_gradients = sim.set(sim.gradient);
  auto points_to_weights = sim.set(sim.weight);
  auto nodes_to_x = sim.get(sim.position);
  auto elems_to_nodes = sim.elems_to_nodes();
  auto elems_to_time_len = sim.set(sim.time_step_length);
  auto elems_to_visc_len = sim.set(sim.viscosity_length);
  auto functor = OMEGA_H_LAMBDA(int elem) {
    auto elem_nodes = getnodes<Elem>(elems_to_nodes, elem);
    auto x = getvecs<Elem>(nodes_to_x, elem_nodes);
    auto shape = Elem::shape(x);
    elems_to_time_len[elem] = shape.lengths.time_step_length;
    elems_to_visc_len[elem] = shape.lengths.viscosity_length;
    for (int elem_pt = 0; elem_pt < Elem::points; ++elem_pt) {
      auto pt = elem * Elem::points + elem_pt;
      setgrads<Elem>(points_to_gradients, pt,
          shape.basis_gradients[elem_pt]);
      points_to_weights[pt] = shape.weights[elem_pt];
    }
  };
  parallel_for("config init kernel", sim.elems(), std::move(functor));
}

template <class Elem>
void lump_masses(Simulation& sim) {
  LGR_SCOPE(sim);
  auto points_to_rho = sim.get(sim.density);
  auto points_to_w = sim.get(sim.weight);
  auto nodes_to_elems = sim.nodes_to_elems();
  auto nodes_to_mass = sim.set(sim.nodal_mass);
  auto functor = OMEGA_H_LAMBDA(int node) {
    double node_mass = 0.0;
    for (auto node_elem = nodes_to_elems.a2ab[node];
        node_elem < nodes_to_elems.a2ab[node + 1];
        ++node_elem) {
      auto elem = nodes_to_elems.ab2b[node_elem];
      auto code = nodes_to_elems.codes[node_elem];
      auto elem_node = Omega_h::code_which_down(code);
      double elem_mass = 0.0;
      for (int elem_pt = 0; elem_pt < Elem::points; ++elem_pt) {
        auto point = elem * Elem::points + elem_pt;
        auto rho = points_to_rho[point];
        auto w = points_to_w[point];
        elem_mass += rho * w;
      }
      node_mass += elem_mass * Elem::lumping_factor(elem_node);
    }
    nodes_to_mass[node] = node_mass;
  };
  parallel_for("mass lumping kernel", sim.nodes(), std::move(functor));
}

template <class Elem>
void update_position(Simulation& sim) {
  LGR_SCOPE(sim);
  auto nodes_to_v = sim.getset(sim.velocity);
  auto nodes_to_x = sim.getset(sim.position);
  auto nodes_to_a = sim.get(sim.acceleration);
  auto dt = sim.dt;
  auto functor = OMEGA_H_LAMBDA(int node) {
    auto v_n = getvec<Elem>(nodes_to_v, node);
    auto x_n = getvec<Elem>(nodes_to_x, node);
    auto a_n = getvec<Elem>(nodes_to_a, node);
    auto v_np12 = v_n + (dt / 2.0) * a_n;
    auto x_np1 = x_n + dt * v_np12;
    setvec<Elem>(nodes_to_v, node, v_np12);
    setvec<Elem>(nodes_to_x, node, x_np1);
  };
  parallel_for("position update kernel", sim.nodes(), std::move(functor));
}

template <class Elem>
void update_configuration(Simulation& sim) {
  LGR_SCOPE(sim);
  auto elems_to_nodes = sim.disc.ents_to_nodes(ELEMS);
  auto nodes_to_x = sim.get(sim.position);
  auto points_to_gradients = sim.set(sim.gradient);
  auto points_to_weights = sim.getset(sim.weight);
  auto points_to_rho = sim.getset(sim.density);
  auto elems_to_time_len = sim.set(sim.time_step_length);
  auto elems_to_visc_len = sim.set(sim.viscosity_length);
  auto functor = OMEGA_H_LAMBDA(int elem) {
    auto elem_nodes = getnodes<Elem>(elems_to_nodes, elem);
    auto x = getvecs<Elem>(nodes_to_x, elem_nodes);
    auto shape = Elem::shape(x);
    elems_to_time_len[elem] = shape.lengths.time_step_length;
    elems_to_visc_len[elem] = shape.lengths.viscosity_length;
    for (int elem_pt = 0; elem_pt < Elem::points; ++elem_pt) {
      auto pt = elem * Elem::points + elem_pt;
      setgrads<Elem>(points_to_gradients, pt,
          shape.basis_gradients[elem_pt]);
      auto w_n = points_to_weights[pt];
      auto rho_n = points_to_rho[pt];
      auto m = w_n * rho_n;
      auto w_np1 = shape.weights[elem_pt];
      auto rho_np1 = m / w_np1;
      points_to_weights[pt] = w_np1;
      points_to_rho[pt] = rho_np1;
    }
  };
  parallel_for("config update kernel", sim.elems(), std::move(functor));
}

template <class Elem>
void correct_velocity(Simulation& sim) {
  LGR_SCOPE(sim);
  auto nodes_to_v = sim.getset(sim.velocity);
  auto nodes_to_a = sim.get(sim.acceleration);
  auto dt = sim.dt;
  auto functor = OMEGA_H_LAMBDA(int node) {
    auto v_np12 = getvec<Elem>(nodes_to_v, node);
    auto a_np1 = getvec<Elem>(nodes_to_a, node);
    auto v_np1 = v_np12 + (dt / 2.0) * a_np1;
    setvec<Elem>(nodes_to_v, node, v_np1);
  };
  parallel_for("velocity correction kernel", sim.nodes(), std::move(functor));
}

template <class Elem>
void compute_stress_divergence(Simulation& sim) {
  LGR_SCOPE(sim);
  auto points_to_sigma = sim.get(sim.stress);
  auto points_to_grads = sim.get(sim.gradient);
  auto points_to_weights = sim.set(sim.weight);
  auto nodes_to_f = sim.set(sim.force);
  auto nodes_to_elems = sim.nodes_to_elems();
  auto functor = OMEGA_H_LAMBDA(int node) {
    auto node_f = zero_vector<Elem::dim>();
    for (auto node_elem = nodes_to_elems.a2ab[node];
        node_elem < nodes_to_elems.a2ab[node + 1]; ++node_elem) {
      auto elem = nodes_to_elems.ab2b[node_elem];
      auto code = nodes_to_elems.codes[node_elem];
      auto elem_node = Omega_h::code_which_down(code);
      for (int elem_pt = 0; elem_pt < Elem::points; ++elem_pt) {
        auto point = elem * Elem::points + elem_pt;
        auto grad = getvec<Elem>(points_to_grads, point * Elem::nodes + elem_node);
        auto sigma = getsymm<Elem>(points_to_sigma, point);
        auto weight = points_to_weights[point];
        auto cell_f = - (sigma * grad) * weight;
        node_f += cell_f;
      }
      setvec<Elem>(nodes_to_f, node, node_f);
    }
    setvec<Elem>(nodes_to_f, node, node_f);
  };
  parallel_for("stress divergence kernel", sim.nodes(), std::move(functor));
}

template <class Elem>
void compute_nodal_acceleration(Simulation& sim) {
  LGR_SCOPE(sim);
  auto nodes_to_f = sim.get(sim.force);
  auto nodes_to_m = sim.get(sim.nodal_mass);
  auto nodes_to_a = sim.set(sim.acceleration);
  auto functor = OMEGA_H_LAMBDA(int node) {
    auto f = getvec<Elem>(nodes_to_f, node);
    auto m = nodes_to_m[node];
    auto a = f / m;
    setvec<Elem>(nodes_to_a, node, a);
  };
  parallel_for("nodal acceleration kernel", sim.nodes(), std::move(functor));
}

void apply_tractions(Simulation& /*sim*/) {
//if (!sim.has(sim.traction)) return;
//apply_conditions(sim, sim.traction);
}

template <class Elem>
void compute_point_time_steps(Simulation& sim) {
  LGR_SCOPE(sim);
  auto points_to_c = sim.get(sim.wave_speed);
  auto elems_to_h = sim.get(sim.time_step_length);
  auto points_to_dt = sim.set(sim.point_time_step);
  double max = std::numeric_limits<double>::max();
  auto functor = OMEGA_H_LAMBDA(int point) {
    auto elem = point / Elem::points;
    auto h = elems_to_h[elem];
    auto c = points_to_c[point];
    auto dt = (c == 0.0) ? max : (h / c);
    points_to_dt[point] = dt;
  };
  parallel_for("time delta kernel", sim.points(), std::move(functor));
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
