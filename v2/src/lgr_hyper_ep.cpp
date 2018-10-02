#include <lgr_hyper_ep.hpp>
#include <lgr_simulation.hpp>
#include <lgr_for.hpp>
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
    Teuchos::ParameterList& params,
    Properties& props,
    Elastic& elastic)
{
  // Set the defaults
  elastic = Elastic::LINEAR_ELASTIC;
  // Elastic model
  if (!params.isSublist("elastic")) {
    Omega_h_fail("elastic submodel must be defined");
  }
  auto pl = params.sublist("elastic");
  if (pl.isParameter("hyperelastic")) {
    std::string hyperelastic = pl.get<std::string>("hyperelastic");
    if (hyperelastic == "neo hookean") {
      elastic = hyper_ep::Elastic::NEO_HOOKEAN;
    } else {
      std::ostringstream os;
      os << "Hyper elastic model \""<< hyperelastic << "\" not recognized";
      auto str = os.str();
      Omega_h_fail("%s\n", str.c_str());
    }
  }
  if (!pl.isParameter("E")) {
    Omega_h_fail("Young's modulus \"E\" modulus must be defined");
  }
  double E = pl.get<double>("E");
  if (E <= 0.) {
    Omega_h_fail("Young's modulus \"E\" must be positive");
  }
  if (!pl.isParameter("Nu")) {
    Omega_h_fail("Poisson's ratio \"Nu\" must be defined");
  }
  double Nu = pl.get<double>("Nu");
  if (Nu <= -1. || Nu >= .5) {
    Omega_h_fail("Invalid value for Poisson's ratio \"Nu\"");
  }
  props.youngs_modulus = E;
  props.poissons_ratio = Nu;
}

void read_and_validate_plastic_params(
    Teuchos::ParameterList& params,
    Properties& props,
    Hardening& hardening,
    RateDependence& rate_dep)
{
  // Set the defaults
  hardening = Hardening::NONE;
  rate_dep = RateDependence::NONE;
  double max_double = std::numeric_limits<double>::max();
  props.yield_strength = max_double; // Yield strength
  props.hardening_modulus = 0.; // Hardening modulus
  props.hardening_exponent = 1.; // Power law hardening exponent
  props.c1 = 298.; // The rest of the properties are model dependent
  props.c2 = 0.;
  props.c3 = 0.;
  props.c4 = 0.;
  props.ep_dot_0 = 0.;
  if (!params.isSublist("plastic")) {
    return;
  }
  auto& pl = params.sublist("plastic");
  props.yield_strength =
    pl.get<double>("A", props.yield_strength);
  if (!pl.isParameter("hardening")) {
    hardening = Hardening::NONE;
  } else {
    std::string model = pl.get<std::string>("hardening");
    if (model == "linear isotropic") {
      // Linear isotropic hardening J2 plasticity
      hardening = Hardening::LINEAR_ISOTROPIC;
      props.hardening_modulus =
        pl.get<double>("B", props.hardening_modulus);
    } else if (model == "power law") {
      // Power law hardening
      hardening = Hardening::POWER_LAW;
      props.hardening_modulus =
        pl.get<double>("B", props.hardening_modulus);
      props.hardening_exponent =
        pl.get<double>("N", props.hardening_exponent);
    } else if (model == "zerilli armstrong") {
      // Zerilli Armstrong hardening
      hardening = Hardening::ZERILLI_ARMSTRONG;
      props.hardening_modulus =
        pl.get<double>("B", props.hardening_modulus);
      props.hardening_exponent =
        pl.get<double>("N", props.hardening_exponent);
      props.c1 = pl.get<double>("C1", 0.0);
      props.c2 = pl.get<double>("C2", 0.0);
      props.c3 = pl.get<double>("C3", 0.0);
    } else if (model == "johnson cook") {
      // Johnson Cook hardening
      hardening = Hardening::JOHNSON_COOK;
      props.hardening_modulus =
        pl.get<double>("B", props.hardening_modulus);
      props.hardening_exponent =
        pl.get<double>("N", props.hardening_exponent);
      props.c1 = pl.get<double>("T0", props.c1);
      props.c2 = pl.get<double>("TM", props.c2);
      props.c3 = pl.get<double>("M", props.c3);
    } else {
      std::ostringstream os;
      os << "Unrecognized hardening model \"" << model << "\"";
      auto str = os.str();
      Omega_h_fail("%s\n", str.c_str());
    }
  }
  if (pl.isSublist("rate dependent")) {
    // Rate dependence
    auto& p = pl.sublist("rate dependent");
    auto const type = p.get<std::string>("type", "None");
    if (type == "johnson cook") {
      if (hardening != Hardening::JOHNSON_COOK) {
        Omega_h_fail("johnson cook rate dependent model requires johnson cook hardening");
      }
      rate_dep = RateDependence::JOHNSON_COOK;
      props.c4 = p.get<double>("C", props.c4);
      props.ep_dot_0 = p.get<double>("EPDOT0", props.ep_dot_0);
    } else if (type == "zerilli armstrong") {
      if (hardening != Hardening::ZERILLI_ARMSTRONG) {
        Omega_h_fail("zerilli armstrong rate dependent model requires zerilli armstrong hardening");
      }
      rate_dep = RateDependence::ZERILLI_ARMSTRONG;
      props.c4 = p.get<double>("C4", 0.0);
    } else if (type != "None") {
      std::ostringstream os;
      os << "Unrecognized rate dependent type \"" << type << "\"";
      auto str = os.str();
      Omega_h_fail("%s\n", str.c_str());
    }
  }
}

} // hyper_ep


template <class Elem>
struct HyperEP : public Model<Elem>
{
  // Constant model parameters
  hyper_ep::Elastic elastic_;
  hyper_ep::Hardening hardening_;
  hyper_ep::RateDependence rate_dep_;

  hyper_ep::Properties properties;

  // State dependent variables
  FieldIndex equivalent_plastic_strain;
  FieldIndex equivalent_plastic_strain_rate;  // equivalent plastic strain rate
  FieldIndex defgrad_p;  // plastic part of deformation gradient

  // Kinematics
  FieldIndex defgrad;

  HyperEP(Simulation& sim_in, Teuchos::ParameterList& params) :
    Model<Elem>(sim_in, params)
  {
    elastic_ = hyper_ep::Elastic::LINEAR_ELASTIC;
    hardening_ = hyper_ep::Hardening::NONE;
    rate_dep_ = hyper_ep::RateDependence::NONE;
    // Read input parameters
    // Elastic model
    hyper_ep::read_and_validate_elastic_params(
        params, this->properties, elastic_);
    // Plastic model
    hyper_ep::read_and_validate_plastic_params(
        params, this->properties, hardening_, rate_dep_);
    // Problem dimension
    constexpr auto dim = Elem::dim;
    // Define state dependent variables
    this->equivalent_plastic_strain =
      this->point_define("ep", "equivalent plastic strain", 1, "0");
    this->equivalent_plastic_strain_rate =
      this->point_define("ep_dot", "equivalent plastic strain rate", 1, "0");
    this->defgrad_p =
      this->point_define("Fp", "plastic deformation gradient", square(dim), "I");
    // Define kinematic quantities
    this->defgrad = this->point_define("F", "deformation gradient", square(dim), "I");
  }

  std::uint64_t exec_stages() override final { return AT_MATERIAL_MODEL; }

  char const* name() override final { return "hyper elastic-plastic"; }

  void at_material_model() override final {
    using hyper_ep::tensor_type;
    auto elastic = this->elastic_;
    auto hardening = this->hardening_;
    auto rate_dep = this->rate_dep_;
    //  properties
    auto points_to_rho = this->points_get(this->sim.density);
    // State dependent variables
    auto points_to_ep = this->points_getset(this->equivalent_plastic_strain);
    auto points_to_epdot = this->points_getset(this->equivalent_plastic_strain_rate);
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
      auto Fp = resize<3>(getfull<Elem>(points_to_fp, point));
      // Update the material response
      tensor_type T;  // stress tensor
      double c;
      auto err_c = hyper_ep::update(elastic, hardening, rate_dep,
          props, rho, F, dt, temp, T, c, Fp, ep, epdot);
      if(err_c != hyper_ep::ErrorCode::SUCCESS)
        Omega_h_fail("Failed to update stress tensor");
      // Update in/output variables
      setsymm<Elem>(points_to_stress, point, resize<Elem::dim>(T));
      points_to_wave_speed[point] = c;
      points_to_ep[point] = ep;
      points_to_epdot[point] = epdot;
      setfull<Elem>(points_to_F, point, resize<Elem::dim>(F));
      setfull<Elem>(points_to_fp, point, resize<Elem::dim>(Fp));
    };
    parallel_for("hyper ep kernel", this->points(), std::move(functor));
  }
};

template <class Elem>
ModelBase* hyper_ep_factory(Simulation& sim, std::string const&, Teuchos::ParameterList& pl) {
  return new HyperEP<Elem>(sim, pl);
}

#define LGR_EXPL_INST(Elem) \
template ModelBase* hyper_ep_factory<Elem>(Simulation&, std::string const&, Teuchos::ParameterList&);
LGR_EXPL_INST_ELEMS
#undef LGR_EXPL_INST

}
