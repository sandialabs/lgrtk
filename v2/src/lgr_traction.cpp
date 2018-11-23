#include <lgr_traction.hpp>

namespace lgr {

template <class Elem>
void apply_tractions_elem(Simulation& sim) {
  using Side = typename Elem::side;
  auto const subset = sim.fields[sim.traction].support->subset;
  if (subset->count() == 0) return;
  apply_conditions(sim, sim.traction);
  auto const nodes_to_sides = sim.disc.nodes_to_ents(SIDES);
  auto const traction_points_to_traction = sim.get(sim.traction);
  auto const traction_points_to_weights = sim.get(sim.traction_weight);
  auto const nodes_to_force = sim.getset(sim.force);
  auto functor = OMEGA_H_LAMBDA(int const node) {
    auto const begin = nodes_to_sides.a2ab[node];
    auto const end = nodes_to_sides.a2ab[node + 1];
    auto node_force = getvec<Elem>(nodes_to_force, node);
    for (auto const node_side = begin; node_side < end; ++node_side) {
      auto const side = nodes_to_sides.ab2b[node_side];
      auto const traction_side = sides_to_traction_sides[side];
      if (traction_side == -1) continue;
      auto const code = nodes_to_sides.codes[node_side];
      auto const side_node = Omega_h::code_which_down(code);
      for (int side_point = 0; side_point < Side::points; ++side_point) {
        auto const traction_point = traction_side * Side::points + side_point;
        auto const traction = getvec<Elem>(traction_points_to_traction, traction_point);
        auto const weight = traction_points_to_weights[traction_point * Side::nodes + side_node];
        node_force += traction * weight;
      }
    }
    setvec<Elem>(nodes_to_force, node_force);
  };
  parallel_for(sim.disc.count(NODES), std::move(functor));
}

}
