#include <lgr_adapt.hpp>
#include <lgr_simulation.hpp>
#include <lgr_for.hpp>
#include <Omega_h_stack.hpp>
#include <Omega_h_metric.hpp>

namespace lgr {

Adapter::Adapter(Simulation& sim_in)
  :sim(sim_in) {
}

void Adapter::setup(Teuchos::ParameterList& pl) {
  should_adapt = pl.isSublist("adapt");
  if (should_adapt) {
    opts = decltype(opts)(&sim.disc.mesh);
    auto& adapt_pl = pl.sublist("adapt");
    opts.min_quality_desired =
      adapt_pl.get<double>("desired quality",
          sim.dim() == 3 ? 0.3 : 0.4);
    trigger_quality =
      adapt_pl.get<double>("trigger quality", opts.min_quality_desired - 0.02);
    trigger_length_ratio =
      adapt_pl.get<double>("trigger length ratio",
          opts.max_length_allowed * 0.9);
    minimum_length =
      adapt_pl.get<double>("minimum length", 0.0);
    auto verbosity = adapt_pl.get<std::string>("verbosity", "each adapt");
    if (verbosity == "each adapt") {
      opts.verbosity = Omega_h::EACH_ADAPT;
    } else if (verbosity == "each rebuild") {
      opts.verbosity = Omega_h::EACH_REBUILD;
    } else if (verbosity == "extra stats") {
      opts.verbosity = Omega_h::EXTRA_STATS;
    } else if (verbosity == "silent") {
      opts.verbosity = Omega_h::SILENT;
    }
    this->gradation_rate = adapt_pl.get<double>("gradation rate", 1.0);
    should_coarsen_with_expansion = adapt_pl.get<bool>("coarsen with expansion", false);
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
  if (should_coarsen_with_expansion) coarsen_metric_with_expansion();
  {
    auto metric = sim.disc.mesh.get_array<double>(0, "metric");
    metric = Omega_h::limit_metric_gradation(&sim.disc.mesh, metric, this->gradation_rate);
    sim.disc.mesh.add_tag(0, "metric", 1, metric);
  }
  remap->before_adapt();
  sim.fields.forget_disc();
  sim.subsets.forget_disc();
  Omega_h::adapt(&sim.disc.mesh, opts);
  sim.subsets.learn_disc();
  sim.fields.learn_disc();
  remap->after_adapt();
  return true;
}

void Adapter::coarsen_metric_with_expansion() {
  auto const old_metric = sim.disc.mesh.get_array<double>(0, "metric");
  auto const implied_metric = get_implied_isos(&sim.disc.mesh);
  auto const nverts = sim.disc.mesh.nverts();
  auto const class_dims = sim.disc.mesh.get_array<Omega_h::Byte>(0, "class_dim");
  auto const new_metric = Omega_h::Write<double>(nverts);
  auto const dim = sim.disc.mesh.dim();
  auto functor = OMEGA_H_LAMBDA(int vert) {
    if (class_dims[vert] == Omega_h::Byte(dim)) {
      new_metric[vert] = Omega_h::min2(implied_metric[vert], old_metric[vert]);
    } else {
      new_metric[vert] = old_metric[vert];
    }
  };
  parallel_for("metric expansion kernel", nverts, std::move(functor));
  sim.disc.mesh.add_tag(0, "metric", 1, read(new_metric));
}

}
