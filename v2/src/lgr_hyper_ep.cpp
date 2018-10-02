#include <lgr_hyper_ep.hpp>
#include <lgr_simulation.hpp>
#include <lgr_for.hpp>

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
    std::vector<double>& props,
    Elastic& elastic)
{
  // Set the defaults
  elastic = Elastic::LINEAR_ELASTIC;
  // Elastic model
  if (!params.isSublist("elastic")) {
    Omega_h_fail("elastic submodel must be defined");
  }
  props.resize(2);
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
  props[0] = E;
  props[1] = Nu;
}

void read_and_validate_plastic_params(
    Teuchos::ParameterList& params,
    std::vector<double>& props,
    Hardening& hardening,
    RateDependence& rate_dep)
{
  // Set the defaults
  hardening = Hardening::NONE;
  rate_dep = RateDependence::NONE;
  props.resize(8);
  double max_double = std::numeric_limits<double>::max();
  props[0] = max_double;  // Yield strength
  props[1] = 0.;          // Hardening modulus
  props[2] = 1.;          // Power law hardening exponent
  props[3] = 298.;        // The rest of the properties are model dependent
  props[4] = 0.;
  props[5] = 0.;
  props[6] = 0.;
  props[7] = 0.;
  if (!params.isSublist("plastic")) {
    return;
  }
  auto& pl = params.sublist("plastic");
  if (!pl.isParameter("hardening")) {
    hardening = Hardening::NONE;
    props[0] = pl.get<double>("A", props[0]);
  } else {
    std::string model = pl.get<std::string>("hardening");
    if (model == "linear isotropic") {
      // Linear isotropic hardening J2 plasticity
      hardening = Hardening::LINEAR_ISOTROPIC;
      props[0] = pl.get<double>("A", props[0]);
      props[1] = pl.get<double>("B", props[1]);
    } else if (model == "power law") {
      // Power law hardening
      hardening = Hardening::POWER_LAW;
      props[0] = pl.get<double>("A", props[0]);
      props[1] = pl.get<double>("B", props[1]);
      props[2] = pl.get<double>("N", props[2]);
    } else if (model == "zerilli armstrong") {
      // Zerilli Armstrong hardening
      hardening = Hardening::ZERILLI_ARMSTRONG;
      props[0] = pl.get<double>("A", props[0]);
      props[1] = pl.get<double>("B", props[1]);
      props[2] = pl.get<double>("N", props[2]);
      props[3] = pl.get<double>("C1", 0.0);
      props[4] = pl.get<double>("C2", 0.0);
      props[5] = pl.get<double>("C3", 0.0);
    } else if (model == "johnson cook") {
      // Johnson Cook hardening
      hardening = Hardening::JOHNSON_COOK;
      props[0] = pl.get<double>("A", props[0]);
      props[1] = pl.get<double>("B", props[1]);
      props[2] = pl.get<double>("N", props[2]);
      props[3] = pl.get<double>("T0", props[3]);
      props[4] = pl.get<double>("TM", props[4]);
      props[5] = pl.get<double>("M", props[5]);
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
      props[6] = p.get<double>("C", props[6]);
      props[7] = p.get<double>("EPDOT0", props[7]);
    } else if (type == "zerilli armstrong") {
      if (hardening != Hardening::ZERILLI_ARMSTRONG) {
        Omega_h_fail("zerilli armstrong rate dependent model requires zerilli armstrong hardening");
      }
      rate_dep = RateDependence::ZERILLI_ARMSTRONG;
      props[6] = p.get<double>("C4", 0.0);
    } else if (type != "None") {
      std::ostringstream os;
      os << "Unrecognized rate dependent type \"" << type << "\"";
      auto str = os.str();
      Omega_h_fail("%s\n", str.c_str());
    }
  }
}

} // Hyperepdetails


template <class Elem>
struct HyperEP : public Model<Elem>
{

  // Constant model parameters
  hyper_ep::Elastic elastic_;
  hyper_ep::Hardening hardening_;
  hyper_ep::RateDependence rate_dep_;

  // Field varying model parameters
  FieldIndex E;
  FieldIndex Nu;

  // Plastic hardening and rate dependence
  FieldIndex A;
  FieldIndex B;
  FieldIndex N;
  FieldIndex C1;
  FieldIndex C2;
  FieldIndex C3;
  FieldIndex C4;
  FieldIndex C5;
  FieldIndex M;


  // State dependent variables
  FieldIndex equivalent_plastic_strain;
  FieldIndex equivalent_plastic_strain_rate;  // equivalent plastic strain rate
  FieldIndex defgrad_p;  // plastic part of deformation gradient

  // Kinematics
  FieldIndex defgrad;

  HyperEP(Simulation& sim_in, Teuchos::ParameterList& params) :
    Model<Elem>(sim_in, params)
  {
    using std::to_string;

    elastic_ = hyper_ep::Elastic::LINEAR_ELASTIC;
    hardening_ = hyper_ep::Hardening::NONE;
    rate_dep_ = hyper_ep::RateDependence::NONE;

    // Read input parameters
    std::vector<double> props;

    // Elastic model
    hyper_ep::read_and_validate_elastic_params(params, props, elastic_);
    this->E = this->point_define("E", "Young's modulus", 1, to_string(props[0]));
    this->Nu = this->point_define("Nu", "Poisson's ratio", 1, to_string(props[1]));

    // Plastic model
    hyper_ep::read_and_validate_plastic_params(params, props, hardening_, rate_dep_);
    this->A = this->point_define("A", "A", 1, to_string(props[0]));

    if (hardening_ == hyper_ep::Hardening::NONE) {
      this->B = this->point_define("B", "UNUSED B", 1, to_string(props[1]));
      this->N = this->point_define("N", "UNUSED N", 1, to_string(props[2]));
      this->C1 = this->point_define("C1", "UNUSED C1", 1, to_string(props[3]));
      this->C2 = this->point_define("C2", "UNUSED C2", 1, to_string(props[4]));
      this->C3 = this->point_define("C3", "UNUSED C3", 1, to_string(props[5]));
    }
    else if (hardening_ == hyper_ep::Hardening::LINEAR_ISOTROPIC ||
             hardening_ == hyper_ep::Hardening::POWER_LAW) {
      this->B = (hardening_ == hyper_ep::Hardening::LINEAR_ISOTROPIC) ?
                this->point_define("B", "UNUSED B", 1, to_string(props[1])) :
                this->point_define("B", "B", 1, to_string(props[1]));
      this->N = this->point_define("N", "N", 1, to_string(props[2]));
      this->C1 = this->point_define("C1", "UNUSED C1", 1, to_string(props[3]));
      this->C2 = this->point_define("C2", "UNUSED C2", 1, to_string(props[4]));
      this->C3 = this->point_define("C3", "UNUSED C3", 1, to_string(props[5]));
    }
    else if (hardening_ == hyper_ep::Hardening::JOHNSON_COOK) {
      this->B = this->point_define("B", "B", 1, to_string(props[1]));
      this->N = this->point_define("N", "N", 1, to_string(props[2]));
      this->C1 = this->point_define("T0", "reference temperature", 1, to_string(props[3]));
      this->C2 = this->point_define("TM", "melt temperature", 1, to_string(props[4]));
      this->C3 = this->point_define("M", "M", 1, to_string(props[5]));
    }
    else if (hardening_ == hyper_ep::Hardening::ZERILLI_ARMSTRONG) {
      this->B = this->point_define("B", "B", 1, to_string(props[1]));
      this->N = this->point_define("N", "N", 1, to_string(props[2]));
      this->C1 = this->point_define("C1", "C1", 1, to_string(props[3]));
      this->C2 = this->point_define("C2", "C2", 1, to_string(props[4]));
      this->C3 = this->point_define("C3", "C3", 1, to_string(props[5]));
    }

    if (rate_dep_ == hyper_ep::RateDependence::NONE) {
      this->C4 = this->point_define("C4", "UNUSED C4", 1, to_string(props[7]));
      this->C5 = this->point_define("C5", "UNUSED C5", 1, to_string(props[7]));
    }
    else if (rate_dep_ == hyper_ep::RateDependence::JOHNSON_COOK) {
      this->C4 = this->point_define("C", "C", 1, to_string(props[6]));
      this->C5 = this->point_define("EPDOT0", "EPDOT0", 1, to_string(props[7]));
    }
    else if (rate_dep_ == hyper_ep::RateDependence::ZERILLI_ARMSTRONG) {
      this->C4 = this->point_define("C4", "C4", 1, to_string(props[6]));
      this->C5 = this->point_define("C5", "UNUSED C5", 1, to_string(props[7]));
    }

    // Problem dimension
    constexpr auto dim = Elem::dim;

    // Define state dependent variables
    this->equivalent_plastic_strain =
      this->point_define("ep", "equivalent plastic strain", 1, "0");
    this->equivalent_plastic_strain_rate =
      this->point_define("epdot", "rate of equivalent plastic strain", 1, "0");
    this->defgrad_p =
      this->point_define("Fp", "plastic part of deformation gradient", square(dim), "I");

    // Define kinematic quantities
    this->defgrad = this->point_define("F", "deformation gradient", square(dim), "I");
  }

  std::uint64_t exec_stages() override final { return AT_MATERIAL_MODEL; }

  char const* name() override final { return "hyper elastic-plastic"; }

  void at_material_model() override final {
    using tensor_type = Matrix<3,3>;

    auto elastic = this->elastic_;
    auto hardening = this->hardening_;
    auto rate_dep = this->rate_dep_;

    //  properties
    auto points_to_rho = this->points_get(this->sim.density);
    auto points_to_emod = this->points_get(this->E);
    auto points_to_nu = this->points_get(this->Nu);

    // Plastic hardening
    auto points_to_a = this->points_get(this->A);
    auto points_to_b = this->points_get(this->B);
    auto points_to_n = this->points_get(this->N);

    auto points_to_c1 = this->points_get(this->C1);
    auto points_to_c2 = this->points_get(this->C2);
    auto points_to_c3 = this->points_get(this->C3);
    auto points_to_c4 = this->points_get(this->C4);
    auto points_to_c5 = this->points_get(this->C5);

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

    auto functor = OMEGA_H_LAMBDA(int point) {
      auto elem = point / Elem::points;
      auto elem_nodes = getnodes<Elem>(elems_to_nodes, elem);
      auto v = getvecs<Elem>(nodes_to_v, elem_nodes);
      auto dN_dxnp1 = getgrads<Elem>(points_to_grad, point);
      auto dv_dxnp1 = grad<Elem>(dN_dxnp1, v);
      auto I = identity_matrix<Elem::dim, Elem::dim>();
      auto dxn_dxnp1 = I - dt * dv_dxnp1;
      auto dxnp1_dxn = invert(dxn_dxnp1);
      auto dxn_dX = getfull<Elem>(points_to_F, point);
      auto dxnp1_dX = dxn_dX * dxnp1_dxn;
      setfull<Elem>(points_to_F, point, dxnp1_dX);

      //  properties
      std::vector<double> props {points_to_emod[point],
                                 points_to_nu[point],
                                 points_to_a[point],
                                 points_to_b[point],
                                 points_to_n[point],
                                 points_to_c1[point],
                                 points_to_c2[point],
                                 points_to_c3[point],
                                 points_to_c4[point],
                                 points_to_c5[point]};

      tensor_type T;  // stress tensor
      auto F = resize<3>(dxnp1_dX);
      auto rho = points_to_rho[point];
      double c;
      double temp = 0.;  // FIXME

      // State dependent variables
      auto ep = points_to_ep[point];
      auto epdot = points_to_epdot[point];
      auto Fp = resize<3>(getfull<Elem>(points_to_fp, point));

      // Update the material response
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
