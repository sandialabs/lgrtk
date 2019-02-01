#include <lgr_indset.hpp>
#include <lgr_for.hpp>
#include <lgr_model.hpp>
#include <lgr_simulation.hpp>
#include <Omega_h_indset.hpp>
#include <Omega_h_random.hpp>

namespace lgr {

template <class Elem>
struct Indset : public Model<Elem> {
  using Model<Elem>::sim;
  FieldIndex independent_set;
  Indset(Simulation& sim_in)
      : Model<Elem>(
            sim_in, sim_in.fields[sim_in.fields.find("independent set")]
                        .class_names),
        independent_set(sim_in.fields.find("independent set")) {}
  std::uint64_t exec_stages() override final { return BEFORE_FIELD_UPDATE; }
  char const* name() override final { return "independent set"; }
  void before_field_update() override final {
    if (sim.has(independent_set)) return;
    auto const vert_globals = sim.disc.mesh.globals(0);
    Omega_h::I64 const seed = 0;
    Omega_h::I64 const counter = 0;
    auto const vert_randoms =
      Omega_h::unit_uniform_random_reals_from_globals(
          vert_globals, seed, counter);
    auto const nverts = vert_globals.size();
    auto const candidates = Omega_h::Bytes(nverts, Omega_h::Byte(1));
    auto const indset_bytes = Omega_h::find_indset(
        &sim.disc.mesh, 0, vert_randoms, candidates);
    auto const nodes_to_indset = sim.set(this->independent_set);
    OMEGA_H_CHECK(sim.nodes() == nverts);
    OMEGA_H_CHECK(nverts == nodes_to_indset.size());
    auto functor = OMEGA_H_LAMBDA(int const node) {
      nodes_to_indset[node] = (indset_bytes[node] == Omega_h::Byte(1)) ? 1.0 : 0.0;
    };
    parallel_for(nverts, std::move(functor));
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

