#include <lgr_scalar.hpp>
#include <lgr_l2_error.hpp>
#include <Omega_h_expr.hpp>
#include <Omega_h_reduce.hpp>
#include <Omega_h_int_iterator.hpp>
#include <lgr_simulation.hpp>

namespace lgr {

using Omega_h::transform_reduce;
using Omega_h::plus;
using II = Omega_h::IntIterator;

struct L2Error : public Scalar {
  Omega_h::ExprEnv env;
  std::shared_ptr<Omega_h::ExprOp> op;
  FieldIndex field_index;
  FieldIndex expected_field_index;
  L2Error(Simulation& sim_in, std::string const& name_in,
      Omega_h::InputMap& pl)
    :Scalar(sim_in, name_in)
  {
    auto field_name = pl.get<std::string>("field");
    field_index = sim.fields.find(field_name);
    auto& field = sim.fields[field_index];
    auto long_name = std::string("expected ") + field.long_name;
    expected_field_index = sim.fields.define(
        long_name, long_name, field.ncomps, field.support);
    auto& expected_field = sim.fields[expected_field_index];
    expected_field.finalize_definition(sim.supports);
    auto expr = pl.get<std::string>("expected value");
    expected_field.conditions.push_back(Condition(
          &expected_field, sim, expr, expected_field.support, never()));
  }
  double compute_value() override {
    auto& expected_field = sim.fields[expected_field_index];
    auto node_coords = sim.get(sim.position);
    expected_field.conditions[0].apply(sim.time, node_coords, sim.fields);
    auto support = expected_field.support;
    auto& field = sim.fields[field_index];
    auto computed_data = Omega_h::read(field.storage);
    auto expected_data = Omega_h::read(expected_field.storage);
    if (field.entity_type == NODES) {
      support = sim.supports.get_support(ELEMS, true, support->subset->class_names);
      computed_data = support->interpolate_nodal(field.ncomps, computed_data);
      expected_data = support->interpolate_nodal(field.ncomps, expected_data);
    }
#define LGR_EXPL_INST(Elem) \
    if (sim.elem_name == Elem::name()) { \
      return compute_integral<Elem>(computed_data, expected_data, support); \
    }
    LGR_EXPL_INST_ELEMS
#undef LGR_EXPL_INST
    OMEGA_H_NORETURN(0.0);
  }
  template <class Elem>
  double compute_integral(Omega_h::Read<double> computed_data,
      Omega_h::Read<double> expected_data, Support* support) {
    auto& field = sim.fields[field_index];
    auto weights = sim.points_get<Elem>(sim.weight, support->subset);
    int ncomps = field.ncomps;
    auto transform = OMEGA_H_LAMBDA(int point) -> double {
      auto w = weights[point];
      double term = 0.0;
      for (int comp = 0; comp < ncomps; ++comp) {
        term += square(computed_data[point * ncomps + comp] - expected_data[point * ncomps + comp]);
      }
      return term * w;
    };
    double const sum_squares = transform_reduce(II(0), II(support->count()), 0.0, plus<double>(), std::move(transform));
    return std::sqrt(sum_squares);
  }
  void out_of_line_virtual_method() override;
};

void L2Error::out_of_line_virtual_method() {}

Scalar* l2_error_factory(Simulation& sim, std::string const& name, Omega_h::InputMap& pl) {
  return new L2Error(sim, name, pl);
}

}
