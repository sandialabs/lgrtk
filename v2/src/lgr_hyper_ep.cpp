#include <lgr_for.hpp>
#include <lgr_hyper_ep.hpp>
#include <lgr_simulation.hpp>
#include <sstream>

namespace lgr {

namespace hyper_ep {

void read_and_validate_elastic_params(
    Omega_h::InputMap& params, Properties& props) {
  // Set the defaults
  props.elastic = Elastic::LINEAR_ELASTIC;
  // Elastic model
  if (!params.is_map("elastic")) {
    Omega_h_fail("elastic submodel must be defined");
  }
  auto& pl = params.get_map("elastic");
  if (pl.is<std::string>("hyperelastic")) {
    auto hyperelastic = pl.get<std::string>("hyperelastic");
    if (hyperelastic == "neo hookean") {
      props.elastic = hyper_ep::Elastic::NEO_HOOKEAN;
    } else if (hyperelastic == "hyperelastic") {
      props.elastic = hyper_ep::Elastic::HYPERELASTIC;
    } else {
      std::ostringstream os;
      os << "Hyper elastic model \"" << hyperelastic << "\" not recognized";
      auto str = os.str();
      Omega_h_fail("%s\n", str.c_str());
    }
  }
  if (!pl.is<double>("E")) {
    Omega_h_fail("Young's modulus \"E\" modulus must be defined");
  }
  double E = pl.get<double>("E");
  if (E <= 0.) {
    Omega_h_fail("Young's modulus \"E\" must be positive");
  }
  if (!pl.is<double>("Nu")) {
    Omega_h_fail("Poisson's ratio \"Nu\" must be defined");
  }
  double Nu = pl.get<double>("Nu");
  if (Nu <= -1. || Nu >= .5) {
    Omega_h_fail("Invalid value for Poisson's ratio \"Nu\"");
  }
  props.E = E;
  props.Nu = Nu;
}

void read_and_validate_plastic_params(
    Omega_h::InputMap& params, Properties& props) {
  // Set the defaults
  props.hardening = Hardening::NONE;
  props.rate_dep = RateDependence::NONE;
  if (!params.is_map("plastic")) {
    return;
  }
  auto& pl = params.get_map("plastic");
  auto max_double_str = std::to_string(std::numeric_limits<double>::max());
  props.p0 = std::numeric_limits<double>::max();
  if (!pl.is_map("hardening")) {
    props.hardening = Hardening::NONE;
    props.p0 = pl.get<double>("Y0", max_double_str.c_str());
  } else {
    auto& p2 = pl.get_map("hardening");
    std::string type = p2.get<std::string>("type", "none");
    if (type == "linear isotropic") {
      // Linear isotropic hardening J2 plasticity
      props.hardening = Hardening::LINEAR_ISOTROPIC;
      props.p0 = p2.get<double>("Y0", max_double_str.c_str());
      props.p1 = p2.get<double>("H", "0.0");
    } else if (type == "power law") {
      // Power law hardening
      props.hardening = Hardening::POWER_LAW;
      props.p0 = p2.get<double>("Y0", max_double_str.c_str());
      props.p1 = p2.get<double>("B", "0.0");
      props.p2 = p2.get<double>("N", "1.0");
    } else if (type == "johnson cook") {
      // Johnson Cook hardening
      props.hardening = Hardening::JOHNSON_COOK;
      props.p0 = p2.get<double>("A", max_double_str.c_str());
      props.p1 = p2.get<double>("B", "0.0");
      props.p2 = p2.get<double>("N", "1.0");
      props.p3 = p2.get<double>("T0", "298.0");
      props.p4 = p2.get<double>("TM", max_double_str.c_str());
      props.p5 = p2.get<double>("M", "0.0");
    } else if (type == "zerilli armstrong") {
      // Zerilli Armstrong hardening
      props.hardening = Hardening::ZERILLI_ARMSTRONG;
      props.p0 = p2.get<double>("A", max_double_str.c_str());
      props.p1 = p2.get<double>("B", "0.0");
      props.p2 = p2.get<double>("N", "1.0");
      props.p3 = p2.get<double>("C", "0.0");
      props.p4 = p2.get<double>("alpha_0", "0.0");
      props.p5 = p2.get<double>("alpha_1", "0.0");
      props.p6 = p2.get<double>("D", "0.0");
      props.p7 = p2.get<double>("beta_0", "0.0");
      props.p8 = p2.get<double>("beta_1", "0.0");
    } else if (type != "none") {
      std::ostringstream os;
      os << "Unrecognized hardening type \"" << type << "\"";
      auto str = os.str();
      Omega_h_fail("%s\n", str.c_str());
    }
  }
  if (pl.is_map("rate dependent")) {
    // Rate dependence
    auto& p2 = pl.get_map("rate dependent");
    auto const type = p2.get<std::string>("type", "none");
    if (type == "johnson cook") {
      if (props.hardening != Hardening::JOHNSON_COOK) {
        Omega_h_fail(
            "johnson cook rate dependent type requires johnson cook hardening");
      }
      props.rate_dep = RateDependence::JOHNSON_COOK;
      props.p6 = p2.get<double>("C", "0.0");
      props.p7 = p2.get<double>("EPDOT0", "0.0");
    } else if (type != "none") {
      std::ostringstream os;
      os << "Unrecognized rate dependent type \"" << type << "\"";
      auto str = os.str();
      Omega_h_fail("%s\n", str.c_str());
    }
  }
}

void read_and_validate_damage_params(
    Omega_h::InputMap& params, Properties& props) {
  // Set the defaults
  props.damage = Damage::NONE;
  if (!params.is_map("plastic")) {
    return;
  }
  auto& pl = params.get_map("plastic");
  auto max_double_str = std::to_string(std::numeric_limits<double>::max());
  if (!pl.is_map("damage")) {
    props.damage = Damage::NONE;
  } else {
    auto& p2 = pl.get_map("damage");
    std::string type = p2.get<std::string>("type", "none");
    if (type == "johnson cook") {
      // Johnson Cook damage
      props.damage = Damage::JOHNSON_COOK;
      props.D1 = p2.get<double>("D1", "0.0");
      props.D2 = p2.get<double>("D2", "0.0");
      props.D3 = p2.get<double>("D3", "0.0");
      props.D4 = p2.get<double>("D4", "0.0");
      props.D5 = p2.get<double>("D5", "0.0");

      // Temperature dependence
      props.D6 = p2.get<double>("T0", "298.0");
      props.D7 = p2.get<double>("TM", max_double_str.c_str());

      props.D8 =
          p2.get<double>("spall failure strain", "0.6");

      props.D0 = p2.get<double>("D0", "0.0");  // Initial scalar damage
      props.DC = p2.get<double>("DC", "0.7"); // Critical scalar damage

      bool no_shear =
          static_cast<bool>(p2.get<double>("allow no shear", "0.0"));
      bool no_tension =
          static_cast<bool>(p2.get<double>("allow no tension", "0.0"));
      bool zero_stress =
          static_cast<bool>(p2.get<double>("set stress to zero", "0.0"));

      if (!(no_shear && no_tension && zero_stress)) {
        // by default, allow no tension
        no_tension = true;
      }
      props.allow_no_shear = no_shear;
      props.allow_no_tension = no_tension;
      props.set_stress_to_zero = zero_stress;


    } else if (type != "none") {
      std::ostringstream os;
      os << "Unrecognized damage type \"" << type << "\"";
      auto str = os.str();
      Omega_h_fail("%s\n", str.c_str());
    }
  }
}

inline void read_and_validate_eos_params(
    Omega_h::InputMap& params, Properties& props) {
  // Set the defaults

  props.eos = EOS::NONE;
  if (!params.is_map("equation of state"))
    return;
  auto& pl = params.get_map("equation of state");

  std::string type = pl.get<std::string>("type", "none");
  if (type == "Mie-Gruneisen") {
    props.eos = EOS::MIE_GRUNEISEN;

    if (!pl.is<double>("initial density"))
      Omega_h_fail("Mie Gruneisen EOS initial density must be defined");
    props.rho0 = pl.get<double>("initial density");
    if (props.rho0 <= 0.)
      Omega_h_fail("Mie Gruneisen EOS initial density must be positive");

    if (!pl.is<double>("Gruneisen parameter"))
      Omega_h_fail("Mie Gruneisen EOS Gruneisen parameter must be defined");
    props.gamma0 = pl.get<double>("Gruneisen parameter");
    if (props.gamma0 <= 0.)
      Omega_h_fail("Mie Gruneisen EOS Gruneisen parameter must be positive");

    if (!pl.is<double>("unshocked sound speed"))
      Omega_h_fail("Mie Gruneisen EOS unshocked sound speed must be defined");
    props.cs = pl.get<double>("unshocked sound speed");
    if (props.cs <= 0.)
      Omega_h_fail("Mie Gruneisen EOS unshocked sound speed must be positive");

    if (!pl.is<double>("Us/Up ratio"))
      Omega_h_fail("Mie Gruneisen EOS Us/Up ratio must be defined");
    props.s1 = pl.get<double>("Us/Up ratio");
    if (props.s1 <= 0.)
      Omega_h_fail("Mie Gruneisen EOS Us/Up ratio must be positive");

    props.e0 = pl.get<double>("specific internal energy", "1");

  } else if (type != "none") {
    std::ostringstream os;
    os << "Unrecognized equation of state type \"" << type << "\".  ";
    os << "Valid equations of state are: Mie-Gruneisen.";
    auto str = os.str();
    Omega_h_fail("%s\n", str.c_str());
  }
}

}  // namespace hyper_ep

template <class Elem>
struct HyperEP : public Model<Elem> {
  // Constant model parameters
  hyper_ep::Properties properties;

  // State dependent variables
  FieldIndex effective_bulk_modulus;
  FieldIndex equivalent_plastic_strain;
  FieldIndex equivalent_plastic_strain_rate;  // equivalent plastic strain rate
  FieldIndex localized_;
  FieldIndex scalar_damage;
  FieldIndex defgrad_p;  // plastic part of deformation gradient
  FieldIndex defgrad_n;

  // Kinematics
  FieldIndex defgrad;

  HyperEP(Simulation& sim_in, Omega_h::InputMap& params)
      : Model<Elem>(sim_in, params) {
    // Read input parameters
    // Elastic model
    hyper_ep::read_and_validate_elastic_params(params, this->properties);
    // Plastic model
    hyper_ep::read_and_validate_plastic_params(params, this->properties);
    // Damage model
    hyper_ep::read_and_validate_damage_params(params, this->properties);
    // Equation of state
    hyper_ep::read_and_validate_eos_params(params, this->properties);
    // Problem dimension
    constexpr auto dim = Elem::dim;

    // Define state dependent variables
    this->equivalent_plastic_strain =
        this->point_define("ep", "equivalent plastic strain", 1, "0");
    this->equivalent_plastic_strain_rate =
        this->point_define("ep_dot", "equivalent plastic strain rate", 1, "0");
    this->scalar_damage = this->point_define("dp", "scalar damage", 1, "0");
    this->localized_ = this->point_define("localized", "localized", 1, "0");
    this->defgrad_p = this->point_define(
        "Fp", "plastic deformation gradient", square(dim), "I");
    // Define kinematic quantities
    this->defgrad =
        this->point_define("F", "deformation gradient", square(dim), "I");
    this->effective_bulk_modulus = this->point_define(
        "kappa_tilde", "effective bulk modulus", 1, RemapType::NONE, params, "");
  }

  std::uint64_t exec_stages() override final { return AT_MATERIAL_MODEL; }

  char const* name() override final { return "hyper elastic-plastic"; }

  void at_material_model() override final {
    //  properties
    auto points_to_rho = this->points_get(this->sim.density);
    // State dependent variables
    auto points_to_ep = this->points_getset(this->equivalent_plastic_strain);
    auto points_to_epdot =
        this->points_getset(this->equivalent_plastic_strain_rate);
    auto points_to_dp = this->points_getset(this->scalar_damage);
    auto points_to_localized = this->points_getset(this->localized_);
    auto points_to_fp = this->points_getset(this->defgrad_p);
    auto points_to_F = this->points_get(this->defgrad);
    // Variables to update
    auto points_to_stress = this->points_set(this->sim.stress);
    auto points_to_wave_speed = this->points_set(this->sim.wave_speed);
    auto points_to_kappa_tilde = this->points_set(this->effective_bulk_modulus);
    // Kinematics
    auto dt = this->sim.dt;
    auto nodes_to_v = this->sim.get(this->sim.velocity);
    auto points_to_grad = this->points_get(this->sim.gradient);
    auto elems_to_nodes = this->get_elems_to_nodes();
    auto props = this->properties;
    auto functor = OMEGA_H_LAMBDA(int const point) {
      auto const dxnp1_dX = getfull<Elem>(points_to_F, point);
      auto const F = resize<3>(dxnp1_dX);
      auto const rho = points_to_rho[point];
      double const temp = 0.;  // FIXME
      // State dependent variables
      auto ep = points_to_ep[point];
      auto epdot = points_to_epdot[point];
      auto dp = points_to_dp[point];
      auto localized = points_to_localized[point];
      auto Fp = resize<3>(getfull<Elem>(points_to_fp, point));
      // Update the material response
      Tensor<3> T;  // stress tensor
      double c;
      hyper_ep::update(props, rho, F, dt, temp, T, c, Fp, ep, epdot, dp, localized);
      // Update in/output variables
      setstress(points_to_stress, point, T);
      points_to_wave_speed[point] = c;
      points_to_ep[point] = ep;
      points_to_epdot[point] = epdot;
      points_to_dp[point] = dp;
      points_to_localized[point] = localized;
      points_to_kappa_tilde[point] =
        3.0 * (rho * c * c) * (1.0 - props.Nu) / (1.0 + props.Nu);
      setfull<Elem>(points_to_fp, point, resize<Elem::dim>(Fp));
    };
    parallel_for(this->points(), std::move(functor));
  }
};

template<class Elem>
ModelBase* hyper_ep_factory(
    Simulation& sim, std::string const&, Omega_h::InputMap& pl)
{
  return new HyperEP<Elem>(sim, pl);
}

#define LGR_EXPL_INST(Elem)                                                    \
		template ModelBase* hyper_ep_factory<Elem>(                               \
				Simulation&, std::string const&, Omega_h::InputMap&);
LGR_EXPL_INST_ELEMS
#undef LGR_EXPL_INST

}  // namespace lgr
