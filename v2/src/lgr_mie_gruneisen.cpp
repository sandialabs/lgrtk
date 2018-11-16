#include <lgr_mie_gruneisen.hpp>
#include <lgr_scope.hpp>
#include <lgr_simulation.hpp>
#include <lgr_for.hpp>

namespace lgr {

template <class Elem>
struct MieGruneisen : public Model<Elem> {

  FieldIndex rho0_;
  FieldIndex gamma0_;
  FieldIndex cs_;
  FieldIndex s1_;
  FieldIndex specific_internal_energy;

  MieGruneisen(Simulation& sim_in, Omega_h::InputMap& pl) :
    Model<Elem>(sim_in, pl)
  {
    this->rho0_ = this->point_define(
        "rho_0", "initial density", 1, "");
    this->gamma0_ = this->point_define(
        "gamma_0", "Gruneisen parameter", 1, "");
    this->cs_ = this->point_define(
        "c_0", "unshocked sound speed", 1, "");
    this->s1_ = this->point_define(
        "S1", "Us/Up ratio", 1, "");
    this->specific_internal_energy =
      this->point_define("e", "specific internal energy", 1,
          RemapType::PER_UNIT_MASS, "");
  }

  std::uint64_t exec_stages() override final { return AT_MATERIAL_MODEL; }

  char const* name() override final { return "Mie-Gruniesen"; }

  void at_material_model() override final {
    auto points_to_rho = this->points_get(this->sim.density);

    auto points_to_e = this->points_get(this->specific_internal_energy);
    auto points_to_rho0 = this->points_get(this->rho0_);
    auto points_to_gamma0 = this->points_get(this->gamma0_);
    auto points_to_cs = this->points_get(this->cs_);
    auto points_to_s1 = this->points_get(this->s1_);

    auto points_to_sigma = this->points_set(this->sim.stress);
    auto points_to_c = this->points_set(this->sim.wave_speed);
    auto functor = OMEGA_H_LAMBDA(int point) {
      auto const rho_np1 = points_to_rho[point];
      auto const e_np1_est = points_to_e[point];
      auto const rho0 = points_to_rho0[point];
      auto const gamma0 = points_to_gamma0[point];
      auto const c0 = points_to_cs[point];
      auto const s1 = points_to_s1[point];
      double c;
      double pressure;
      mie_gruneisen_update(rho0, gamma0, c0, s1, rho_np1, e_np1_est, pressure, c);
      auto sigma = diagonal(fill_vector<Elem::dim>(-pressure));
      setsymm<Elem>(points_to_sigma, point, sigma);
      points_to_c[point] = c;
    };
    parallel_for(this->points(), std::move(functor));
  }
};

template <class Elem>
ModelBase* mie_gruneisen_factory(Simulation& sim, std::string const&, Omega_h::InputMap& pl) {
  return new MieGruneisen<Elem>(sim, pl);
}

#define LGR_EXPL_INST(Elem) \
template ModelBase* mie_gruneisen_factory<Elem>(Simulation&, std::string const&, Omega_h::InputMap&);
LGR_EXPL_INST_ELEMS
#undef LGR_EXPL_INST

}
