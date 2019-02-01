#include <lgr_indset.hpp>
#include <lgr_for.hpp>
#include <lgr_model.hpp>
#include <lgr_simulation.hpp>
#include <Omega_h_indset.hpp>
#include <Omega_h_random.hpp>

// DEBUG!
#include <Omega_h_file.hpp>

namespace lgr {

template <class Elem>
struct Indset : public Model<Elem> {
  using Model<Elem>::sim;
  Indset(Simulation& sim_in)
      : Model<Elem>(
            sim_in, sim_in.disc.covering_class_names())
  {}
  std::uint64_t exec_stages() override final { return BEFORE_FIELD_UPDATE; }
  char const* name() override final { return "independent set"; }
  void before_field_update() override final {
    if (sim.disc.mesh.has_tag(1, "LGR independent set")) return;
//  auto const edge_globals = sim.disc.mesh.globals(1);
//  Omega_h::I64 const seed = 0;
//  Omega_h::I64 const counter = 0;
//  auto const edge_qualities =
//    Omega_h::unit_uniform_random_reals_from_globals(
//        edge_globals, seed, counter);
    auto const edge_qualities_w = Omega_h::Write<double>(sim.disc.mesh.nents(1));
    auto const coords = sim.disc.mesh.coords();
    auto const edges_to_nodes = sim.disc.mesh.ask_verts_of(1);
    auto functor = OMEGA_H_LAMBDA(int const edge) {
      auto const node0 = edges_to_nodes[edge * 2 + 0];
      auto const node1 = edges_to_nodes[edge * 2 + 1];
      auto const a = Omega_h::get_vector<Elem::dim>(coords, node0);
      auto const b = Omega_h::get_vector<Elem::dim>(coords, node1);
      auto const dir = normalize(b - a);
      edge_qualities_w[edge] = std::abs(dir(0));
    };
    parallel_for(sim.disc.mesh.nents(1), std::move(functor));
    auto const edge_qualities = read(edge_qualities_w);
    auto const nedges = edge_qualities.size();
    auto const candidates = Omega_h::Bytes(nedges, Omega_h::Byte(1));
    auto const indset_bytes = Omega_h::find_indset(
        &sim.disc.mesh, 1, edge_qualities, candidates);
    sim.disc.mesh.add_tag(1, "LGR independent set", 1, indset_bytes);
    Omega_h::vtk::write_vtu("indset.vtu", &sim.disc.mesh, 1);
  }
};

template <class Elem>
ModelBase* independent_set_factory(
    Simulation& sim, std::string const&, Omega_h::InputMap&) {
  return new Indset<Elem>(sim);
}

#define LGR_EXPL_INST(Elem)                                                    \
  template ModelBase* independent_set_factory<Elem>(                      \
      Simulation&, std::string const&, Omega_h::InputMap&);
LGR_EXPL_INST_ELEMS
#undef LGR_EXPL_INST

}  // namespace lgr

