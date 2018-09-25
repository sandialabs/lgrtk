#include <lgr_hyper_ep.hpp>
#include <lgr_simulation.hpp>
#include <lgr_for.hpp>

namespace lgr {

namespace HyperEPDetails {

std::string
get_error_code_string(const ErrorCode& code) {
  std::string error_code_string{"UNKNOWN"};
  switch (code) {
    case ErrorCode::NOT_SET:
      error_code_string = "NOT SET";
      break;
    case ErrorCode::SUCCESS:
      error_code_string = "SUCCESS";
      break;
    case ErrorCode::LINEAR_ELASTIC_FAILURE:
      error_code_string = "LINEAR ELASTIC FAILURE";
      break;
    case ErrorCode::HYPERELASTIC_FAILURE:
      error_code_string = "HYPERELASTIC FAILURE";
      break;
    case ErrorCode::RADIAL_RETURN_FAILURE:
      error_code_string = "RADIAL RETURN FAILURE";
      break;
    case ErrorCode::ELASTIC_DEFORMATION_UPDATE_FAILURE:
      error_code_string = "ELASTIC DEFORMATION UPDATE FAILURE";
      break;
    case ErrorCode::MODEL_EVAL_FAILURE:
      error_code_string = "MODEL EVAL FAILURE";
      break;
  }
  return error_code_string;
}

} // Hyperepdetails


template <class Elem>
struct HyperEP : public Model<Elem>
{

  // Constant model parameters
  HyperEPDetails::Elastic elastic_;
  HyperEPDetails::Hardening hardening_;
  HyperEPDetails::RateDependence rate_dep_;

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

    elastic_ = HyperEPDetails::Elastic::LINEAR_ELASTIC;
    hardening_ = HyperEPDetails::Hardening::NONE;
    rate_dep_ = HyperEPDetails::RateDependence::NONE;

    // Read input parameters
    std::vector<double> props;

    // Elastic model
    HyperEPDetails::read_and_validate_elastic_params(params, props, elastic_);
    this->E = this->point_define("E", "Young's modulus", 1, to_string(props[0]));
    this->Nu = this->point_define("Nu", "Poisson's ratio", 1, to_string(props[1]));

    // Plastic model
    HyperEPDetails::read_and_validate_plastic_params(params, props, hardening_, rate_dep_);
    this->A = this->point_define("A", "A", 1, to_string(props[0]));

    if (hardening_ == HyperEPDetails::Hardening::NONE) {
      this->B = this->point_define("B", "UNUSED B", 1, to_string(props[1]));
      this->N = this->point_define("N", "UNUSED N", 1, to_string(props[2]));
      this->C1 = this->point_define("C1", "UNUSED C1", 1, to_string(props[3]));
      this->C2 = this->point_define("C2", "UNUSED C2", 1, to_string(props[4]));
      this->C3 = this->point_define("C3", "UNUSED C3", 1, to_string(props[5]));
    }
    else if (hardening_ == HyperEPDetails::Hardening::LINEAR_ISOTROPIC ||
             hardening_ == HyperEPDetails::Hardening::POWER_LAW) {
      this->B = (hardening_ == HyperEPDetails::Hardening::LINEAR_ISOTROPIC) ?
                this->point_define("B", "UNUSED B", 1, to_string(props[1])) :
                this->point_define("B", "B", 1, to_string(props[1]));
      this->N = this->point_define("N", "N", 1, to_string(props[2]));
      this->C1 = this->point_define("C1", "UNUSED C1", 1, to_string(props[3]));
      this->C2 = this->point_define("C2", "UNUSED C2", 1, to_string(props[4]));
      this->C3 = this->point_define("C3", "UNUSED C3", 1, to_string(props[5]));
    }
    else if (hardening_ == HyperEPDetails::Hardening::JOHNSON_COOK) {
      this->B = this->point_define("B", "B", 1, to_string(props[1]));
      this->N = this->point_define("N", "N", 1, to_string(props[2]));
      this->C1 = this->point_define("T0", "reference temperature", 1, to_string(props[3]));
      this->C2 = this->point_define("TM", "melt temperature", 1, to_string(props[4]));
      this->C3 = this->point_define("M", "M", 1, to_string(props[5]));
    }
    else if (hardening_ == HyperEPDetails::Hardening::ZERILLI_ARMSTRONG) {
      this->B = this->point_define("B", "B", 1, to_string(props[1]));
      this->N = this->point_define("N", "N", 1, to_string(props[2]));
      this->C1 = this->point_define("C1", "C1", 1, to_string(props[3]));
      this->C2 = this->point_define("C2", "C2", 1, to_string(props[4]));
      this->C3 = this->point_define("C3", "C3", 1, to_string(props[5]));
    }

    if (rate_dep_ == HyperEPDetails::RateDependence::NONE) {
      this->C4 = this->point_define("C4", "UNUSED C4", 1, to_string(props[7]));
      this->C5 = this->point_define("C5", "UNUSED C5", 1, to_string(props[7]));
    }
    else if (rate_dep_ == HyperEPDetails::RateDependence::JOHNSON_COOK) {
      this->C4 = this->point_define("C", "C", 1, to_string(props[6]));
      this->C5 = this->point_define("EPDOT0", "EPDOT0", 1, to_string(props[7]));
    }
    else if (rate_dep_ == HyperEPDetails::RateDependence::ZERILLI_ARMSTRONG) {
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

  void update_state() override final {
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
      auto err_c = HyperEPDetails::eval(elastic, hardening, rate_dep,
          props, rho, F, dt, temp, T, c, Fp, ep, epdot);
      if(err_c != HyperEPDetails::ErrorCode::SUCCESS)
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
