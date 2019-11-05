#include <lgr_for.hpp>
#include <lgr_sierra_J2.hpp>
#include <lgr_simulation.hpp>

namespace lgr {

template <class Elem>
struct SierraJ2 : public Model<Elem>
{
  FieldIndex poissons_ratio;
  FieldIndex elastic_modulus;
  FieldIndex hardening_modulus;
  FieldIndex hardening_exponent;
  FieldIndex yield_strength;
  FieldIndex plastic_def_grad;
  FieldIndex equivalent_plastic_strain;

  SierraJ2(Simulation& sim_in, Omega_h::InputMap& pl) : Model<Elem>(sim_in, pl)
  {
    this->poissons_ratio =
        this->point_define("nu", "Poissons ratio", 1, RemapType::PICK, pl, "");
    this->elastic_modulus =
        this->point_define("E", "elastic modulus", 1, RemapType::PICK, pl, "");
    this->hardening_modulus = this->point_define(
        "K", "hardening modulus", 1, RemapType::PICK, pl, "");
    this->hardening_exponent = this->point_define(
        "beta", "hardening exponent", 1, RemapType::PICK, pl, "");
    this->yield_strength =
        this->point_define("Y", "yield strength", 1, RemapType::PICK, pl, "");
    this->plastic_def_grad = this->point_define(
        "Fp",
        "plastic deformation gradient",
        square(this->sim.dim()),
        RemapType::PICK,
        pl,
        "I");
    this->equivalent_plastic_strain = this->point_define(
        "eqps", "equivalent plastic strain", 1, RemapType::PICK, pl, "0.0");
  }

  std::uint64_t
  exec_stages() override final
  {
    return AT_MATERIAL_MODEL;
  }
  char const*
  name() override final
  {
    return "Sierra J2";
  }

  void
  at_material_model() override final
  {
    // local vars
    auto points_to_nu   = this->points_get(this->poissons_ratio);
    auto points_to_E    = this->points_get(this->elastic_modulus);
    auto points_to_K    = this->points_get(this->hardening_modulus);
    auto points_to_beta = this->points_get(this->hardening_exponent);
    auto points_to_Y    = this->points_get(this->yield_strength);
    auto points_to_Fp   = this->points_getset(this->plastic_def_grad);
    auto points_to_eqps = this->points_getset(this->equivalent_plastic_strain);

    // simulation vars
    auto points_to_rho    = this->points_get(this->sim.density);
    auto points_to_F      = this->points_get(this->sim.def_grad);
    auto points_to_F_init = this->points_get(this->sim.def_grad_init);
    auto points_to_stress = this->points_set(this->sim.stress);
    auto points_to_c      = this->points_set(this->sim.wave_speed);

    auto functor = OMEGA_H_LAMBDA(int point)
    {
      // get the total deformation gradient in 3x3 form
      auto F_small      = getfull<Elem>(points_to_F, point);
      auto F_init_small = getfull<Elem>(points_to_F_init, point);
      auto F_tot_small  = F_small * F_init_small;
      auto F            = identity_tensor<3>();
      for (int i = 0; i < Elem::dim; ++i)
        for (int j = 0; j < Elem::dim; ++j) F(i, j) = F_tot_small(i, j);

      // get the old plastic history variables
      auto eqps     = points_to_eqps[point];
      auto Fp_small = getfull<Elem>(points_to_Fp, point);
      auto Fp       = identity_tensor<3>();
      for (int i = 0; i < Elem::dim; ++i)
        for (int j = 0; j < Elem::dim; ++j) Fp(i, j) = Fp_small(i, j);

      // get the material properties
      auto rho  = points_to_rho[point];
      auto E    = points_to_E[point];
      auto nu   = points_to_nu[point];
      auto K    = points_to_K[point];
      auto beta = points_to_beta[point];
      auto Y    = points_to_Y[point];

      // values to update
      double    c;
      Tensor<3> sigma;

      // update the stress, wave speed, and plastic history vars
      sierra_J2_update(rho, E, nu, K, beta, Y, F, Fp, eqps, sigma, c);

      // resize Fp to its small value
      Tensor<Elem::dim> Fp_new;
      for (int i = 0; i < Elem::dim; ++i)
        for (int j = 0; j < Elem::dim; ++j) Fp_new(i, j) = Fp(i, j);

      // set new values
      setstress(points_to_stress, point, sigma);
      setfull(points_to_Fp, point, Fp_new);
      points_to_eqps[point] = eqps;
      points_to_c[point]    = c;
    };
    parallel_for(this->points(), std::move(functor));
  }
};

template <class Elem>
ModelBase*
sierra_J2_factory(Simulation& sim, std::string const&, Omega_h::InputMap& pl)
{
  return new SierraJ2<Elem>(sim, pl);
}

#define LGR_EXPL_INST(Elem)                    \
  template ModelBase* sierra_J2_factory<Elem>( \
      Simulation&, std::string const&, Omega_h::InputMap&);
LGR_EXPL_INST_ELEMS
#undef LGR_EXPL_INST

}  // namespace lgr
