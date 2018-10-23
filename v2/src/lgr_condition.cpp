#include <lgr_condition.hpp>
#include <lgr_field.hpp>
#include <lgr_support.hpp>
#include <lgr_subsets.hpp>
#include <lgr_disc.hpp>
#include <lgr_supports.hpp>
#include <lgr_when.hpp>
#include <lgr_riemann.hpp>
#include <Omega_h_map.hpp>
#include <Omega_h_scalar.hpp>
#include <Omega_h_math_lang.hpp>

namespace lgr {

using Omega_h::divide_no_remainder;

// yes, its wasted effort to recompute all three fields
// and throw away the other two each time one field is
// asked for
#define RIEMANN_EXPR(var) \
static Omega_h::any riemann_expr_##var( \
    std::vector<Omega_h::any>& args) { \
  auto left_density = Omega_h::any_cast<double>(args.at(0)); \
  auto right_density = Omega_h::any_cast<double>(args.at(1)); \
  auto left_pressure = Omega_h::any_cast<double>(args.at(2)); \
  auto right_pressure = Omega_h::any_cast<double>(args.at(3)); \
  auto shock_x = Omega_h::any_cast<double>(args.at(4)); \
  auto gamma = Omega_h::any_cast<double>(args.at(5)); \
  auto t = Omega_h::any_cast<double>(args.at(6)); \
  auto x = Omega_h::any_cast<Omega_h::Reals>(args.at(7)); \
  auto result = exact_riemann( \
      left_density, right_density, \
      left_pressure, right_pressure, \
      shock_x, gamma, t, x); \
  return result.var; \
}
RIEMANN_EXPR(velocity)
RIEMANN_EXPR(density)
RIEMANN_EXPR(pressure)

void Condition::init(Supports& supports) {
  auto vars_used = Omega_h::math_lang::get_symbols_used(str);
  uses_old_vals = (vars_used.count(field->short_name) != 0);
  needs_coords = vars_used.count("x");
  needs_reeval = (needs_coords || uses_old_vals || vars_used.count("t"));
  Omega_h::ExprOpsReader reader;
  op = reader.read_ops(str);
  bridge = supports.subsets.get_bridge(support->subset, field->support->subset);
  learn_disc();
  env.register_function("riemann_velocity", riemann_expr_velocity);
  env.register_function("riemann_density", riemann_expr_density);
  env.register_function("riemann_pressure", riemann_expr_pressure);
}

Condition::Condition(Field* field_in, Supports& supports,
    std::string const& str_in, Support* support_in, When* when_in)
:field(field_in)
,str(str_in)
,support(support_in)
,when(when_in)
{
  init(supports);
}

Condition::Condition(Field* field_in, Supports& supports, Omega_h::InputMap& pl)
:field(field_in)
{
  str = pl.get<std::string>("value");
  if (pl.is_list("sets")) {
    auto& class_names_teuchos = pl.get_list("sets");
    ClassNames class_names;
    for (int i = 0; i < class_names_teuchos.size(); ++i) {
      class_names.insert(class_names_teuchos.get<std::string>(i));
    }
    support = supports.get_support(field_in->entity_type, field_in->on_points, class_names);
  } else {
    support = field->support;
  }
  when.reset(setup_when(pl));
  init(supports);
}

void Condition::forget_disc() {
  env = decltype(env)();
  cached_values = decltype(cached_values)();
}

void Condition::learn_disc() {
  env = decltype(env)(support->count(), support->subset->disc.dim());
}

double Condition::next_event(double time) { return when->next_event(time); }

void Condition::apply(double prev_time, double time,
    Omega_h::Read<double> node_coords) {
  if (!when->active(prev_time, time)) return;
  apply(time, node_coords);
}

void Condition::apply(double time, Omega_h::Read<double> node_coords) {
  OMEGA_H_CHECK(field->storage.exists());
  if (needs_reeval || (!cached_values.exists())) {
    if (needs_coords) {
      Omega_h::Reals coords = support->ask_coords(time, node_coords);
      env.register_variable("x", Omega_h::any(coords));
    }
    env.register_variable("t", Omega_h::any(time));
    if (uses_old_vals) {
      Omega_h::Reals old_field_vals(field->storage);
      if (!(bridge->mapping.is_identity)) {
        old_field_vals = Omega_h::unmap(
            bridge->mapping.things, old_field_vals,
            field->ncomps);
      }
      env.register_variable(field->short_name, Omega_h::any(old_field_vals));
    }
    auto result = op->eval(env);
    env.repeat(result);
    cached_values = Omega_h::any_cast<Omega_h::Reals>(result);
  }
  OMEGA_H_CHECK(cached_values.exists());
  OMEGA_H_CHECK(field->storage.exists());
  if (bridge->mapping.is_identity) {
    if (cached_values.size() != field->storage.size()) {
      Omega_h_fail("Value of condition \"%s\" on field \"%s\" was of the wrong size\n",
          str.c_str(), field->long_name.c_str());
    }
    Omega_h::copy_into(cached_values, field->storage);
  } else {
    auto ncomps = divide_no_remainder(cached_values.size(), bridge->mapping.things.size());
    if (field->storage.size() != ncomps * field->support->count()) {
      Omega_h_fail("Value of condition \"%s\" on subset of field \"%s\" was of the wrong size\n",
          str.c_str(), field->long_name.c_str());
    }
    Omega_h::map_into(cached_values, bridge->mapping.things,
        field->storage, ncomps);
  }
}

}
