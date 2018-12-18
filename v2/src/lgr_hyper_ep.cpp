#include <lgr_for.hpp>
#include <lgr_hyper_ep.hpp>
#include <lgr_simulation.hpp>
#include <sstream>

namespace lgr {

namespace hyper_ep {

char const* get_error_code_string(ErrorCode code) {
  switch (code) {
    case ErrorCode::NOT_SET:
      return "NOT SET";
    case ErrorCode::SUCCESS:
      return "SUCCESS";
    case ErrorCode::LINEAR_ELASTIC_FAILURE:
      return "LINEAR ELASTIC FAILURE";
    case ErrorCode::HYPERELASTIC_FAILURE:
      return "HYPERELASTIC FAILURE";
    case ErrorCode::RADIAL_RETURN_FAILURE:
      return "RADIAL RETURN FAILURE";
    case ErrorCode::ELASTIC_DEFORMATION_UPDATE_FAILURE:
      return "ELASTIC DEFORMATION UPDATE FAILURE";
    case ErrorCode::MODEL_EVAL_FAILURE:
      return "MODEL EVAL FAILURE";
  }
  return "UNKNOWN";
}

void read_and_validate_elastic_params(
    Omega_h::InputMap& params, Properties& props, Elastic& elastic) {
  // Set the defaults
  elastic = Elastic::LINEAR_ELASTIC;
  // Elastic model
  if (!params.is_map("elastic")) {
    Omega_h_fail("elastic submodel must be defined");
  }
  auto& pl = params.get_map("elastic");
  if (pl.is<std::string>("hyperelastic")) {
    auto hyperelastic = pl.get<std::string>("hyperelastic");
    if (hyperelastic == "neo hookean") {
      elastic = hyper_ep::Elastic::NEO_HOOKEAN;
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
  props.youngs_modulus = E;
  props.poissons_ratio = Nu;
}

void read_and_validate_plastic_params(Omega_h::InputMap& params,
    Properties& props, Hardening& hardening, RateDependence& rate_dep) {
  // Set the defaults
  hardening = Hardening::NONE;
  rate_dep = RateDependence::NONE;
  if (!params.is_map("plastic")) {
    return;
  }
  auto& pl = params.get_map("plastic");
  auto max_double_str = std::to_string(std::numeric_limits<double>::max());
  props.yield_strength = pl.get<double>("A", max_double_str.c_str());
  if (!pl.is<std::string>("hardening")) {
    hardening = Hardening::NONE;
  } else {
    std::string model = pl.get<std::string>("hardening");
    if (model == "linear isotropic") {
      // Linear isotropic hardening J2 plasticity
      hardening = Hardening::LINEAR_ISOTROPIC;
      props.hardening_modulus = pl.get<double>("B", "0.0");
    } else if (model == "power law") {
      // Power law hardening
      hardening = Hardening::POWER_LAW;
      props.hardening_modulus = pl.get<double>("B", "0.0");
      props.hardening_exponent = pl.get<double>("N", "1.0");
    } else if (model == "zerilli armstrong") {
      // Zerilli Armstrong hardening
      hardening = Hardening::ZERILLI_ARMSTRONG;
      props.hardening_modulus = pl.get<double>("B", "0.0");
      props.hardening_exponent = pl.get<double>("N", "1.0");
      props.c1 = pl.get<double>("C1", "0.0");
      props.c2 = pl.get<double>("C2", "0.0");
      props.c3 = pl.get<double>("C3", "0.0");
    } else if (model == "johnson cook") {
      // Johnson Cook hardening
      hardening = Hardening::JOHNSON_COOK;
      props.hardening_modulus = pl.get<double>("B", "0.0");
      props.hardening_exponent = pl.get<double>("N", "1.0");
      props.c1 = pl.get<double>("T0", "298.0");
      props.c2 = pl.get<double>("TM", max_double_str.c_str());
      props.c3 = pl.get<double>("M", "0.0");
    } else {
      std::ostringstream os;
      os << "Unrecognized hardening model \"" << model << "\"";
      auto str = os.str();
      Omega_h_fail("%s\n", str.c_str());
    }
  }
  if (pl.is_map("rate dependent")) {
    // Rate dependence
    auto& p = pl.get_map("rate dependent");
    auto const type = p.get<std::string>("type", "None");
    if (type == "johnson cook") {
      if (hardening != Hardening::JOHNSON_COOK) {
        Omega_h_fail(
            "johnson cook rate dependent model requires johnson cook "
            "hardening");
      }
      rate_dep = RateDependence::JOHNSON_COOK;
      props.c4 = p.get<double>("C", "0.0");
      props.ep_dot_0 = p.get<double>("EPDOT0", "0.0");
    } else if (type == "zerilli armstrong") {
      if (hardening != Hardening::ZERILLI_ARMSTRONG) {
        Omega_h_fail(
            "zerilli armstrong rate dependent model requires zerilli armstrong "
            "hardening");
      }
      rate_dep = RateDependence::ZERILLI_ARMSTRONG;
      props.c4 = p.get<double>("C4", "0.0");
    } else if (type != "None") {
      std::ostringstream os;
      os << "Unrecognized rate dependent type \"" << type << "\"";
      auto str = os.str();
      Omega_h_fail("%s\n", str.c_str());
    }
  }
}

void read_and_validate_damage_params(
    Omega_h::InputMap& params, Properties& props, Damage& damage) {
  // Set the defaults
  damage = Damage::NONE;
  if (!params.is_map("plastic")) {
    return;
  }
  auto& pl = params.get_map("plastic");
  auto max_double_str = std::to_string(std::numeric_limits<double>::max());
  if (!pl.is<std::string>("damage")) {
    damage = Damage::NONE;
  } else {
    std::string model = pl.get<std::string>("damage");
    if (model == "johnson cook") {
      // Johnson Cook hardening
      damage = Damage::JOHNSON_COOK;
      props.D1 = pl.get<double>("D1", "0.0");
      props.D2 = pl.get<double>("D2", "0.0");
      props.D3 = pl.get<double>("D3", "0.0");
      props.D4 = pl.get<double>("D4", "0.0");
      props.D5 = pl.get<double>("D5", "0.0");
      props.D0 = pl.get<double>("D0", "0.0");  // Initial scalar damage
      props.Dc = pl.get<double>("Dc", "0.7");  // Critical scalar damage
      bool no_shear =
          static_cast<bool>(pl.get<double>("allow no shear", "0.0"));
      bool no_tension =
          static_cast<bool>(pl.get<double>("allow no tension", "0.0"));
      bool zero_stress =
          static_cast<bool>(pl.get<double>("set stress to zero", "0.0"));
      if (!(no_shear && no_tension && zero_stress)) {
        // by default, allow no tension
        no_tension = true;
      }
      props.allow_no_shear = no_shear;
      props.allow_no_tension = no_tension;
      props.set_stress_to_zero = zero_stress;

      // Same as JC hardening.
      props.c1 = pl.get<double>("T0", "298.0");
      props.c2 = pl.get<double>("TM", max_double_str.c_str());
    } else {
      std::ostringstream os;
      os << "Unrecognized damage model \"" << model << "\"";
      auto str = os.str();
      Omega_h_fail("%s\n", str.c_str());
    }
  }
}

}  // namespace hyper_ep

template <class Elem>
struct HyperEP : public Model<Elem> {
  // Constant model parameters
  hyper_ep::Elastic elastic_;
  hyper_ep::Hardening hardening_;
  hyper_ep::RateDependence rate_dep_;
  hyper_ep::Damage damage_;

  hyper_ep::Properties properties;

  // State dependent variables
  FieldIndex equivalent_plastic_strain;
  FieldIndex equivalent_plastic_strain_rate;  // equivalent plastic strain rate
  FieldIndex localized_;
  FieldIndex scalar_damage;
  FieldIndex defgrad_p;  // plastic part of deformation gradient

  // Kinematics
  FieldIndex defgrad;

  HyperEP(Simulation& sim_in, Omega_h::InputMap& params)
      : Model<Elem>(sim_in, params) {
    elastic_ = hyper_ep::Elastic::LINEAR_ELASTIC;
    hardening_ = hyper_ep::Hardening::NONE;
    rate_dep_ = hyper_ep::RateDependence::NONE;
    damage_ = hyper_ep::Damage::NONE;
    // Read input parameters
    // Elastic model
    hyper_ep::read_and_validate_elastic_params(
        params, this->properties, elastic_);
    // Plastic model
    hyper_ep::read_and_validate_plastic_params(
        params, this->properties, hardening_, rate_dep_);
    // Damage model
    hyper_ep::read_and_validate_damage_params(
        params, this->properties, damage_);
    // Problem dimension
    constexpr auto dim = Elem::dim;
    // Define state dependent variables
    this->equivalent_plastic_strain =
        this->point_define("ep", "equivalent plastic strain", 1, "0");
    this->equivalent_plastic_strain_rate =
        this->point_define("ep_dot", "equivalent plastic strain rate", 1, "0");
    this->scalar_damage = this->point_define("dp", "scalar damage", 1, "0");
    this->localized_ =
        this->point_define("localized", "localization has occurred", 1, "0");
    this->defgrad_p = this->point_define(
        "Fp", "plastic deformation gradient", square(dim), "I");
    // Define kinematic quantities
    this->defgrad =
        this->point_define("F", "deformation gradient", square(dim), "I");
  }

  std::uint64_t exec_stages() override final { return AT_MATERIAL_MODEL; }

  char const* name() override final { return "hyper elastic-plastic"; }

  void at_material_model() override final {
    using hyper_ep::tensor_type;
    auto elastic = this->elastic_;
    auto hardening = this->hardening_;
    auto rate_dep = this->rate_dep_;
    auto damage = this->damage_;
    //  properties
    auto points_to_rho = this->points_get(this->sim.density);
    // State dependent variables
    auto points_to_ep = this->points_getset(this->equivalent_plastic_strain);
    auto points_to_epdot =
        this->points_getset(this->equivalent_plastic_strain_rate);
    auto points_to_dp = this->points_getset(this->scalar_damage);
    auto points_to_localized = this->points_getset(this->localized_);
    auto points_to_fp = this->points_getset(this->defgrad_p);
    auto points_to_F = this->points_getset(this->defgrad);
    // Variables to update
    auto points_to_stress = this->points_set(this->sim.stress);
    auto points_to_wave_speed = this->points_set(this->sim.wave_speed);
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
      tensor_type T;  // stress tensor
      double c;
      auto err_c = hyper_ep::update(elastic, hardening, rate_dep, damage, props,
          rho, F, dt, temp, T, c, Fp, ep, epdot, dp, localized);
      OMEGA_H_CHECK(err_c == hyper_ep::ErrorCode::SUCCESS);
      // Update in/output variables
      setsymm<Elem>(points_to_stress, point, resize<Elem::dim>(T));
      points_to_wave_speed[point] = c;
      points_to_ep[point] = ep;
      points_to_epdot[point] = epdot;
      points_to_dp[point] = dp;
      points_to_localized[point] = localized;
      setfull<Elem>(points_to_F, point, resize<Elem::dim>(F));
      setfull<Elem>(points_to_fp, point, resize<Elem::dim>(Fp));
    };
    parallel_for("hyper ep kernel", this->points(), std::move(functor));
  }
};

template <class Elem>
ModelBase* hyper_ep_factory(
    Simulation& sim, std::string const&, Omega_h::InputMap& pl) {
  return new HyperEP<Elem>(sim, pl);
}

#define LGR_EXPL_INST(Elem)                                                    \
  template ModelBase* hyper_ep_factory<Elem>(                                  \
      Simulation&, std::string const&, Omega_h::InputMap&);
LGR_EXPL_INST_ELEMS
#undef LGR_EXPL_INST

}  // namespace lgr
