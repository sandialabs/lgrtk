#include <lgr_anti_lock.hpp>
#include <lgr_for.hpp>
#include <lgr_model.hpp>
#include <lgr_simulation.hpp>

namespace lgr {

template <class Elem>
struct AverageJOverPoints : public Model<Elem> {
  FieldIndex deformation_gradient;
  AverageJOverPoints(Simulation& sim_in)
      : Model<Elem>(
            sim_in, sim_in.fields[sim_in.fields.find("deformation gradient")]
                        .class_names),
        deformation_gradient(sim_in.fields.find("deformation gradient")) {}
  std::uint64_t exec_stages() override final { return AFTER_FIELD_UPDATE; }
  char const* name() override final { return "average J over points"; }
  void after_field_update() override final {
    auto const points_to_F = this->points_getset(this->deformation_gradient);
    auto const points_to_w = this->points_get(this->sim.weight);
    auto functor = OMEGA_H_LAMBDA(int const elem) {
      double average_J = 0.0;
      double w_sum = 0.0;
      for (int elem_pt = 0; elem_pt < Elem::points; ++elem_pt) {
        auto const point = elem * Elem::points + elem_pt;
        auto const old_F = getfull<Elem>(points_to_F, point);
        auto const old_J = determinant(old_F);
        auto const w = points_to_w[point];
        average_J += w * old_J;
        w_sum += w;
      }
      average_J /= w_sum;
      for (int elem_pt = 0; elem_pt < Elem::points; ++elem_pt) {
        auto const point = elem * Elem::points + elem_pt;
        auto const old_F = getfull<Elem>(points_to_F, point);
        auto const old_J = determinant(old_F);
        auto const factor = Omega_h::root<Elem::dim>(average_J / old_J);
        auto const new_F = factor * old_F;
        OMEGA_H_CHECK(Omega_h::are_close(average_J, determinant(new_F)));
        setfull<Elem>(points_to_F, point, new_F);
      }
    };
    parallel_for(this->elems(), std::move(functor));
  }
};

template <class Elem>
struct AveragePressureOverPoints : public Model<Elem> {
  AveragePressureOverPoints(Simulation& sim_in)
      : Model<Elem>(
            sim_in, sim_in.fields[sim_in.fields.find("stress")]
                        .class_names)
  {}
  std::uint64_t exec_stages() override final { return BEFORE_SECONDARIES; }
  char const* name() override final { return "average p over points"; }
  void before_secondaries() override final {
    auto const points_to_sigma = this->points_getset(this->sim.stress);
    auto const points_to_w = this->points_get(this->sim.weight);
    auto functor = OMEGA_H_LAMBDA(int const elem) {
      double average_p = 0.0;
      double w_sum = 0.0;
      for (int elem_pt = 0; elem_pt < Elem::points; ++elem_pt) {
        auto const point = elem * Elem::points + elem_pt;
        auto const old_sigma = getstress(points_to_sigma, point);
        auto const old_p = trace(old_sigma) / 3;
        auto const w = points_to_w[point];
        average_p += w * old_p;
        w_sum += w;
      }
      average_p /= w_sum;
      for (int elem_pt = 0; elem_pt < Elem::points; ++elem_pt) {
        auto const point = elem * Elem::points + elem_pt;
        auto const old_sigma = getstress(points_to_sigma, point);
        auto const old_p = trace(old_sigma) / 3;
        auto const factor = average_p - old_p;
        auto const I = identity_matrix<3, 3>();
        auto const new_sigma = old_sigma + I * factor;
        OMEGA_H_CHECK(Omega_h::are_close(average_p, trace(new_sigma) / 3));
        setstress(points_to_sigma, point, new_sigma);
      }
    };
    parallel_for(this->elems(), std::move(functor));
  }
};

template <class Elem>
struct AverageInternalEnergyOverPoints : public Model<Elem> {
  FieldIndex specific_internal_energy;
  AverageInternalEnergyOverPoints(Simulation& sim_in)
      : Model<Elem>(
            sim_in, sim_in.fields[sim_in.fields.find("specific internal energy")]
                        .class_names),
        specific_internal_energy(sim_in.fields.find("specific internal energy")) {}
  std::uint64_t exec_stages() override final { return AFTER_FIELD_UPDATE; }
  char const* name() override final { return "average internal energy over points"; }
  void after_field_update() override final {
    auto const points_to_e = this->points_getset(this->specific_internal_energy);
    auto const points_to_w = this->points_get(this->sim.weight);
    auto const points_to_rho = this->points_get(this->sim.density);
    auto functor = OMEGA_H_LAMBDA(int const elem) {
      double average_e = 0.0;
      double mass_sum = 0.0;
      for (int elem_pt = 0; elem_pt < Elem::points; ++elem_pt) {
        auto const point = elem * Elem::points + elem_pt;
        auto const old_e = points_to_e[point];
        auto const w = points_to_w[point];
        auto const rho = points_to_rho[point];
        average_e += w * rho * old_e;
        mass_sum += w * rho;
      }
      average_e /= mass_sum;
      for (int elem_pt = 0; elem_pt < Elem::points; ++elem_pt) {
        auto const point = elem * Elem::points + elem_pt;
        points_to_e[point] = average_e;
      }
    };
    parallel_for(this->elems(), std::move(functor));
  }
};

template <class Elem>
struct AverageDensityOverPoints : public Model<Elem> {
  AverageDensityOverPoints(Simulation& sim_in)
      : Model<Elem>(
            sim_in, sim_in.fields[sim_in.fields.find("density")]
                        .class_names) {}
  std::uint64_t exec_stages() override final { return BEFORE_FIELD_UPDATE; }
  char const* name() override final { return "average density over points"; }
  void after_field_update() override final {
    auto const points_to_w = this->points_get(this->sim.weight);
    auto const points_to_rho = this->points_getset(this->sim.density);
    auto functor = OMEGA_H_LAMBDA(int const elem) {
      double average_rho = 0.0;
      double w_sum = 0.0;
      for (int elem_pt = 0; elem_pt < Elem::points; ++elem_pt) {
        auto const point = elem * Elem::points + elem_pt;
        auto const w = points_to_w[point];
        auto const old_rho = points_to_rho[point];
        average_rho += w * old_rho;
        w_sum += w;
      }
      average_rho /= w_sum;
      for (int elem_pt = 0; elem_pt < Elem::points; ++elem_pt) {
        auto const point = elem * Elem::points + elem_pt;
        points_to_rho[point] = average_rho;
      }
    };
    parallel_for(this->elems(), std::move(functor));
  }
};

template <class Elem>
struct AverageJOverIndset : public Model<Elem> {
  FieldIndex deformation_gradient;
  FieldIndex independent_set;
  AverageJOverIndset(Simulation& sim_in)
      : Model<Elem>(
            sim_in, sim_in.fields[sim_in.fields.find("deformation gradient")]
                        .class_names),
        deformation_gradient(sim_in.fields.find("deformation gradient"))
  {
    this->independent_set = this->sim.fields.define(
        "i", "independent set", 1, NODES, false, this->sim.disc.covering_class_names());
  }
  std::uint64_t exec_stages() override final { return AFTER_FIELD_UPDATE; }
  char const* name() override final { return "average J over independent set"; }
  void after_field_update() override final {
    auto const points_to_F = this->points_getset(this->deformation_gradient);
    auto const points_to_w = this->points_get(this->sim.weight);
    auto const nodes_to_indset = this->sim.get(this->independent_set);
    auto const nodes_to_elems = this->sim.disc.nodes_to_ents(ELEMS);
    auto functor = OMEGA_H_LAMBDA(int const node) {
      if (nodes_to_indset[node] != 1.0) return;
      auto const begin = nodes_to_elems.a2ab[node];
      auto const end = nodes_to_elems.a2ab[node + 1];
      double average_J = 0.0;
      double w_sum = 0.0;
      for (auto node_elem = begin; node_elem < end; ++node_elem) {
        auto const elem = nodes_to_elems.ab2b[node_elem];
        for (int elem_pt = 0; elem_pt < Elem::points; ++elem_pt) {
          auto const point = elem * Elem::points + elem_pt;
          auto const old_F = getfull<Elem>(points_to_F, point);
          auto const old_J = determinant(old_F);
          auto const w = points_to_w[point];
          average_J += w * old_J;
          w_sum += w;
        }
      }
      average_J /= w_sum;
      for (auto node_elem = begin; node_elem < end; ++node_elem) {
        auto const elem = nodes_to_elems.ab2b[node_elem];
        for (int elem_pt = 0; elem_pt < Elem::points; ++elem_pt) {
          auto const point = elem * Elem::points + elem_pt;
          auto const old_F = getfull<Elem>(points_to_F, point);
          auto const old_J = determinant(old_F);
          auto const factor = Omega_h::root<Elem::dim>(average_J / old_J);
          auto const new_F = factor * old_F;
          OMEGA_H_CHECK(Omega_h::are_close(average_J, determinant(new_F)));
          setfull<Elem>(points_to_F, point, new_F);
        }
      }
    };
    parallel_for(this->sim.nodes(), std::move(functor));
  }
};

template <class Elem>
ModelBase* average_J_over_points_factory(
    Simulation& sim, std::string const&, Omega_h::InputMap&) {
  return new AverageJOverPoints<Elem>(sim);
}

template <class Elem>
ModelBase* average_pressure_over_points_factory(
    Simulation& sim, std::string const&, Omega_h::InputMap&) {
  return new AveragePressureOverPoints<Elem>(sim);
}

template <class Elem>
ModelBase* average_internal_energy_over_points_factory(
    Simulation& sim, std::string const&, Omega_h::InputMap&) {
  return new AverageInternalEnergyOverPoints<Elem>(sim);
}

template <class Elem>
ModelBase* average_density_over_points_factory(
    Simulation& sim, std::string const&, Omega_h::InputMap&) {
  return new AverageDensityOverPoints<Elem>(sim);
}

template <class Elem>
ModelBase* average_J_over_independent_set_factory(
    Simulation& sim, std::string const&, Omega_h::InputMap&) {
  return new AverageJOverIndset<Elem>(sim);
}

#define LGR_EXPL_INST(Elem)                                                    \
  template ModelBase* average_J_over_points_factory<Elem>(                      \
      Simulation&, std::string const&, Omega_h::InputMap&); \
  template ModelBase* average_pressure_over_points_factory<Elem>(                      \
      Simulation&, std::string const&, Omega_h::InputMap&); \
  template ModelBase* average_internal_energy_over_points_factory<Elem>(                      \
      Simulation&, std::string const&, Omega_h::InputMap&); \
  template ModelBase* average_density_over_points_factory<Elem>(                      \
      Simulation&, std::string const&, Omega_h::InputMap&); \
  template ModelBase* average_J_over_independent_set_factory<Elem>(                      \
      Simulation&, std::string const&, Omega_h::InputMap&);
LGR_EXPL_INST_ELEMS
#undef LGR_EXPL_INST

}  // namespace lgr

