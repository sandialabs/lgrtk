#include <lgr_internal_energy.hpp>
#include <lgr_model.hpp>
#include <lgr_simulation.hpp>
#include <lgr_for.hpp>

namespace lgr {

template <class Elem>
struct NodalPressure : public Model<Elem> {
  using Model<Elem>::sim;
  FieldIndex nodal_pressure;
  FieldIndex nodal_pressure_rate;
  NodalPressure(Simulation& sim_in)
    :Model<Elem>(sim_in, sim_in.disc.covering_class_names())
  {  
    auto& everywhere = this->sim.disc.covering_class_names();
    nodal_pressure =
    this->sim.fields.define("p", "nodal pressure", 1, NODES, false, everywhere);
    nodal_pressure_rate =
    this->sim.fields.define("p_dot", "nodal pressure rate", 1, NODES, false, everywhere);
  }
  std::uint64_t exec_stages() override final {
    return BEFORE_MATERIAL_MODEL | BEFORE_SECONDARIES |
      AFTER_CORRECTION | AFTER_MATERIAL_MODEL;
  }
  char const* name() override final { return "nodal pressure"; }
  void before_material_model() override final {
    if (sim.dt == 0.0 && (!sim.fields[nodal_pressure_rate].storage.exists())) {
      zero_nodal_pressure_rate();
    }
    compute_nodal_pressure_predictor();
  }
  void after_material_model() override final {
    auto const points_to_sigma = this->points_getset(sim.stress);
    auto const nodes_to_p = sim.get(nodal_pressure);
    auto const nodes_to_v = sim.get(sim.velocity);
    auto const nodes_to_p_dot = sim.get(nodal_pressure_rate);
    auto const points_to_kappa = this->points_get(effective_bulk_modulus);
    auto const points_to_grad = this->points_get(sim.gradient);
    auto const elems_to_nodes = this->get_elems_to_nodes();
    auto functor = OMEGA_H_LAMBDA(int const point) {
      auto const elem = point/Elem::points;
      auto const elem_point = point%Elem::points;
      auto const elem_nodes = getnodes<Elem>(elems_to_nodes, elem);
      auto const v = getvecs<Elem>(nodes_to_v, elem_nodes);
      auto const grads = getgrads<Elem>(points_to_grad, point);
      auto const grad_v = grad<Elem>(grads, v);
      auto const div_v = trace(grad_v);
      auto const kappa = points_to_kappa[point];
      auto const p_dot = getscals<Elem>(nodes_to_p_dot, elem_nodes);
      auto const basis_values = Elem::basis_values()[elem_point];
      auto const point_p_dot = p_dot * basis_values;
      auto const p_prime = -tau_p * (point_p_dot - kappa * div_v);
      auto const p = getscals<Elem>(nodes_to_p, elem_nodes);
      auto const point_p = p * basis_values;
      auto const sigma = getsymm<Elem>(points_to_sigma, point);
      auto const dev_sigma = deviator(sigma);
      auto const I = identity_matrix<Elem::dim,Elem::dim>();
      auto const sigma_tilde = dev_sigma + I * (point_p + p_prime);
      setsymm<Elem>(points_to_sigma, point, sigma_tilde);
    };
    parallel_for(this->points(), std::move(functor));
  }
  void before_secondaries() override final {
    backtrack_to_midpoint_nodal_pressure();
    zero_nodal_pressure_rate();
  }
  void after_correction() override final {
    correct_nodal_pressure();
  }
  // based on the previous pressure and pressure rate, compute a predicted
  // energy using forward Euler. this predicted pressure is what material models use
  void compute_nodal_pressure_predictor() {
    OMEGA_H_TIME_FUNCTION;
    auto const nodes_to_p = this->sim.getset(this->nodal_pressure);
    auto const nodes_to_p_dot = this->sim.get(this->nodal_pressure_rate);
    auto const dt = this->sim.dt;
    auto functor = OMEGA_H_LAMBDA(int const node) {
      auto const p_dot_n = nodes_to_p_dot[node];
      auto const p_n = nodes_to_p[node];
      auto const p_np1_tilde = p_n + dt * p_dot_n;
      nodes_to_p[node] = p_np1_tilde;
    };
    parallel_for(this->sim.disc.count(NODES), std::move(functor));
  }
  void backtrack_to_midpoint_nodal_pressure() {
    OMEGA_H_TIME_FUNCTION;
    auto const nodes_to_p = this->sim.getset(this->nodal_pressure);
    auto const nodes_to_p_dot = this->sim.get(this->nodal_pressure_rate);
    auto const dt = this->sim.dt;
    auto functor = OMEGA_H_LAMBDA(int const node) {
      auto const p_dot_n = nodes_to_p_dot[node];
      auto const p_np1_tilde = nodes_to_p[node];
      auto const p_np12 = p_np1_tilde - (0.5 * dt) * p_dot_n;
      nodes_to_p[node] = p_np12;
    };
    parallel_for(this->sim.disc.count(NODES), std::move(functor));
  }
  // zero the rate before other models contribute to it
  void zero_nodal_pressure_rate() {
    OMEGA_H_TIME_FUNCTION;
    auto const nodes_to_p_dot = this->sim.set(this->nodal_pressure_rate);
    Omega_h::fill(nodes_to_p_dot, 0.0);
  }
 
  // using the previous midpoint nodal pressure and the current pressure rate,
  // compute the current pressure
  void correct_nodal_pressure() {
    OMEGA_H_TIME_FUNCTION;
    auto const nodes_to_p = this->sim.getset(this->nodal_pressure);
    auto const nodes_to_p_dot = this->sim.get(this->nodal_pressure_rate);
    auto const dt = this->sim.dt;
    auto functor = OMEGA_H_LAMBDA(int const node) {
      auto const p_dot_np1 = nodes_to_p_dot[node];
      auto const p_np12 = nodes_to_p[node];
      auto const p_np1 = p_np12 + (0.5 * dt) * p_dot_np1;
      nodes_to_p[node] = p_np1;
    };
    parallel_for(this->sim.disc.count(NODES), std::move(functor));
  }
};

template <class Elem>
ModelBase* nodal_pressure_factory(Simulation& sim, std::string const&, Omega_h::InputMap&) {
  return new NodalPressure<Elem>(sim);
}

#define LGR_EXPL_INST(Elem) \
template ModelBase* nodal_pressure_factory<Elem>(Simulation&, std::string const&, Omega_h::InputMap&);
LGR_EXPL_INST_ELEMS
#undef LGR_EXPL_INST

}
