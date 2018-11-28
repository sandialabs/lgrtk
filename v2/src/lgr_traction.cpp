#include <lgr_traction.hpp>
#include <lgr_simulation.hpp>
#include <lgr_for.hpp>
#include <lgr_element_functions.hpp>
#include <Omega_h_profile.hpp>
#include <Omega_h_align.hpp>

namespace lgr {

bool has_traction(Simulation& sim) {
  auto const subset = sim.fields[sim.traction].support->subset;
  return !subset->class_names.empty();
}

template <class Elem>
static void eval_traction_weights(Simulation& sim) {
  OMEGA_H_TIME_FUNCTION;
  using Side = typename Elem::side;
  auto const subset = sim.fields[sim.traction].support->subset;
  auto const traction_sides_to_sides = subset->mapping.things;
  auto const traction_points_to_weights = sim.set(sim.traction_weight);
  auto const sides_to_nodes = sim.disc.ents_to_nodes(ELEMS);
  auto const nodes_to_x = sim.get(sim.position);
  auto functor = OMEGA_H_LAMBDA(int const traction_side) {
    auto const side = traction_sides_to_sides[traction_side];
    auto const side_nodes = getnodes<Side>(sides_to_nodes, side);
    auto const x = getvecs<Side>(nodes_to_x, side_nodes);
    auto const w = Side::weights(x);
    for (int side_point = 0; side_point < Side::points; ++side_point) {
      auto const traction_point = traction_side * Side::points + side_point;
      traction_points_to_weights[traction_point] = w[side_point];
    }
  };
  parallel_for(subset->count(), std::move(functor));
}

template <class Elem>
static void integrate_nodal_tractions(Simulation& sim) {
  OMEGA_H_TIME_FUNCTION;
  // TODO: this could be done over only the adjacent nodes
  using Side = typename Elem::side;
  auto const subset = sim.fields[sim.traction].support->subset;
  auto const traction_sides_to_sides = subset->mapping.things;
  auto const sides_to_traction_sides = sim.subsets.acquire_inverse(traction_sides_to_sides, sim.disc.count(SIDES));
  auto const nodes_to_sides = sim.disc.nodes_to_ents(SIDES);
  auto const traction_points_to_traction = sim.get(sim.traction);
  auto const traction_points_to_weights = sim.get(sim.traction_weight);
  auto const nodes_to_force = sim.getset(sim.force);
  auto functor = OMEGA_H_LAMBDA(int const node) {
    auto const begin = nodes_to_sides.a2ab[node];
    auto const end = nodes_to_sides.a2ab[node + 1];
    auto node_force = getvec<Elem>(nodes_to_force, node);
    for (auto node_side = begin; node_side < end; ++node_side) {
      auto const side = nodes_to_sides.ab2b[node_side];
      auto const traction_side = sides_to_traction_sides[side];
      if (traction_side == -1) continue;
      auto const code = nodes_to_sides.codes[node_side];
      auto const side_node = Omega_h::code_which_down(code);
      auto const lumping = Side::lumping(side_node);
      for (int side_point = 0; side_point < Side::points; ++side_point) {
        auto const traction_point = traction_side * Side::points + side_point;
        auto const traction = getvec<Elem>(traction_points_to_traction, traction_point);
        auto const weight = traction_points_to_weights[traction_point * Side::nodes + side_node];
        node_force += traction * weight * lumping;
      }
    }
    setvec<Elem>(nodes_to_force, node, node_force);
  };
  parallel_for(sim.disc.count(NODES), std::move(functor));
  sim.subsets.release_inverse(traction_sides_to_sides);
}

template <class Elem>
static void apply_tractions_tmpl(Simulation& sim) {
  apply_conditions(sim, sim.traction);
  eval_traction_weights<Elem>(sim);
  integrate_nodal_tractions<Elem>(sim);
}

void apply_tractions(Simulation& sim) {
  OMEGA_H_TIME_FUNCTION;
  if (!has_traction(sim)) return;
#define LGR_EXPL_INST(Elem) \
  if (sim.elem_name == Elem::name()) { \
    apply_tractions_tmpl<Elem>(sim); \
  }
LGR_EXPL_INST_ELEMS
#undef LGR_EXPL_INST
}

}
