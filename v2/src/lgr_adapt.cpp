#include <lgr_adapt.hpp>
#include <lgr_simulation.hpp>
#include <Omega_h_stack.hpp>

namespace lgr {

Adapter::Adapter(Simulation& sim_in)
  :sim(sim_in) {
}

void Adapter::setup(Teuchos::ParameterList& pl) {
  should_adapt = pl.isSublist("adapt");
  if (should_adapt) {
    opts = decltype(opts)(&sim.disc.mesh);
    auto& adapt_pl = pl.sublist("adapt");
    trigger_quality =
      adapt_pl.get<double>("trigger quality", opts.min_quality_desired - 0.02);
    trigger_length_ratio =
      adapt_pl.get<double>("trigger length ratio",
          opts.max_length_allowed * 0.9);
    minimum_length =
      adapt_pl.get<double>("minimum length", 0.0);
    opts.verbosity = Omega_h::EACH_ADAPT;
  }
#define LGR_EXPL_INST(Elem) \
  if (sim.elem_name == Elem::name()) { \
    remap.reset(remap_factory<Elem>(sim)); \
    opts.xfer_opts.user_xfer = remap; \
  }
LGR_EXPL_INST_ELEMS
#undef LGR_EXPL_INST
}

bool Adapter::adapt() {
  Omega_h::ScopedTimer timer("lgr::adapt");
  if (!should_adapt) return false;
  sim.disc.mesh.set_coords(sim.get(sim.position)); //linear specific!
  if (!sim.disc.mesh.has_tag(0, "metric")) Omega_h::add_implied_isos_tag(&sim.disc.mesh);
  auto minqual = sim.disc.mesh.min_quality();
  auto maxlen = sim.disc.mesh.max_length();
  if (minqual >= trigger_quality && maxlen <= trigger_length_ratio) return false;
  remap->before_adapt();
  sim.fields.forget_disc();
  sim.subsets.forget_disc();
  Omega_h::adapt(&sim.disc.mesh, opts);
  sim.subsets.learn_disc();
  sim.fields.learn_disc();
  remap->after_adapt();
  return true;
}

}
