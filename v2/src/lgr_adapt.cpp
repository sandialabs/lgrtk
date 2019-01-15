#include <Omega_h_array_ops.hpp>
#include <Omega_h_map.hpp>
#include <Omega_h_metric.hpp>
#include <Omega_h_profile.hpp>
#include <iostream>
#include <lgr_adapt.hpp>
#include <lgr_for.hpp>
#include <lgr_simulation.hpp>

namespace lgr {

Adapter::Adapter(Simulation& sim_in) : sim(sim_in) {}

void Adapter::setup(Omega_h::InputMap& pl) {
  should_adapt = pl.is_map("adapt");
  if (should_adapt) {
    opts = decltype(opts)(&sim.disc.mesh);
    auto& adapt_pl = pl.get_map("adapt");
    auto default_desired_qual = sim.dim() == 3 ? "0.3" : "0.4";
    opts.min_quality_desired =
        adapt_pl.get<double>("desired quality", default_desired_qual);
    auto default_allowed_qual = std::to_string(opts.min_quality_desired - 0.10);
    opts.min_quality_allowed =
        adapt_pl.get<double>("allowed quality", default_allowed_qual.c_str());
    auto default_trigger_qual = std::to_string(opts.min_quality_desired - 0.02);
    trigger_quality =
        adapt_pl.get<double>("trigger quality", default_trigger_qual.c_str());
    trigger_length_ratio = adapt_pl.get<double>("trigger length ratio", "2.1");
    minimum_length = adapt_pl.get<double>("minimum length", "0.0");
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
    if (sim.no_output) opts.verbosity = Omega_h::SILENT;
    this->gradation_rate = adapt_pl.get<double>("gradation rate", "1.0");
    should_coarsen_with_expansion =
        adapt_pl.get<bool>("coarsen with expansion", "false");
#define LGR_EXPL_INST(Elem)                                                    \
  if (sim.elem_name == Elem::name()) {                                         \
    remap.reset(remap_factory<Elem>(sim, pl.get_map("remap")));                                     \
    opts.xfer_opts.user_xfer = remap;                                          \
  }
    LGR_EXPL_INST_ELEMS
#undef LGR_EXPL_INST
    if (!sim.disc.mesh.has_tag(0, "metric"))
      Omega_h::add_implied_isos_tag(&sim.disc.mesh);
    old_quality = sim.disc.mesh.min_quality();
    old_length = sim.disc.mesh.max_length();
  }
}

bool Adapter::adapt() {
  Omega_h::ScopedTimer timer("lgr::adapt");
  if (!should_adapt) return false;
  sim.disc.mesh.set_coords(sim.get(sim.position));  // linear specific!
  if (!sim.disc.mesh.has_tag(0, "metric"))
    Omega_h::add_implied_isos_tag(&sim.disc.mesh);
  auto const minqual = sim.disc.mesh.min_quality();
  auto const maxlen = sim.disc.mesh.max_length();
  auto const is_low_qual = minqual < opts.min_quality_desired;
  auto const is_decreasing_qual = (minqual <= old_quality - 0.02);
  auto const is_really_low_qual = (minqual <= opts.min_quality_allowed + 0.02);
  auto const quality_triggered =
      is_low_qual && (is_decreasing_qual || is_really_low_qual);
  auto const is_long_len = maxlen > opts.max_length_desired;
  auto const is_increasing_len = (maxlen >= old_length + 0.2);
  auto const is_really_long_len = (maxlen >= opts.max_length_allowed - 0.2);
  auto const length_triggered =
      is_long_len && (is_increasing_len || is_really_long_len);
  if ((!quality_triggered) && (!length_triggered)) return false;
  if (should_coarsen_with_expansion) coarsen_metric_with_expansion();
  {
    auto metric = sim.disc.mesh.get_array<double>(0, "metric");
    metric = Omega_h::limit_metric_gradation(
        &sim.disc.mesh, metric, this->gradation_rate, 1e-2, !sim.no_output);
    sim.disc.mesh.add_tag(0, "metric", 1, metric);
  }
  remap->before_adapt();
  sim.fields.forget_disc();
  sim.subsets.forget_disc();
  Omega_h::adapt(&sim.disc.mesh, opts);
  sim.disc.update_from_mesh();
  sim.subsets.learn_disc();
  sim.fields.learn_disc();
  sim.models.learn_disc();
  remap->after_adapt();
  old_quality = sim.disc.mesh.min_quality();
  old_length = sim.disc.mesh.max_length();
  return true;
}

void Adapter::coarsen_metric_with_expansion() {
  OMEGA_H_TIME_FUNCTION;
  auto const old_metric = sim.disc.mesh.get_array<double>(0, "metric");
  auto const implied_metric = get_implied_isos(&sim.disc.mesh);
  auto const nverts = sim.disc.mesh.nverts();
  auto const dim = sim.disc.mesh.dim();
  auto const side_class_dims =
      sim.disc.mesh.get_array<Omega_h::Byte>(dim - 1, "class_dim");
  auto const sides_are_boundaries =
      each_eq_to(side_class_dims, Omega_h::Byte(dim - 1));
  auto const sides_are_outer_boundaries = mark_exposed_sides(&sim.disc.mesh);
  auto const sides_are_inner_boundaries =
      land_each(sides_are_boundaries, invert_marks(sides_are_outer_boundaries));
  auto const verts_are_inner_boundaries =
      mark_down(&sim.disc.mesh, dim - 1, 0, sides_are_inner_boundaries);
  auto const new_metric = Omega_h::Write<double>(nverts);
  auto functor = OMEGA_H_LAMBDA(int vert) {
    if (verts_are_inner_boundaries[vert]) {
      new_metric[vert] = old_metric[vert];
    } else {
      new_metric[vert] = Omega_h::min2(implied_metric[vert], old_metric[vert]);
    }
  };
  parallel_for(nverts, std::move(functor));
  sim.disc.mesh.add_tag(0, "metric", 1, read(new_metric));
}

}  // namespace lgr
