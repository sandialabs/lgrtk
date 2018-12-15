#include <lgr_field.hpp>
#include <lgr_support.hpp>
#include <lgr_supports.hpp>
#include <lgr_subset.hpp>
#include <limits>

namespace lgr {

Field::Field(
    std::string const& short_name_in,
    std::string const& long_name_in,
    int ncomps_in,
    EntityType entity_type_in,
    bool on_points_in,
    ClassNames const& class_names_in,
    bool filling_with_nan_in)
  :short_name(short_name_in)
  ,long_name(long_name_in)
  ,ncomps(ncomps_in)
  ,entity_type(entity_type_in)
  ,on_points(on_points_in)
  ,class_names(class_names_in)
  ,filling_with_nan(filling_with_nan_in)
  ,remap_type(RemapType::NONE)
{
}

bool Field::has() {
  return storage.exists();
}

void Field::ensure_allocated() {
  if (!has()) {
    storage = Omega_h::Write<double>(
        ncomps * support->count(), long_name);
    if (filling_with_nan) {
      auto nan = std::numeric_limits<double>::signaling_NaN();
      Omega_h::fill(storage, nan);
    }
  }
}

Omega_h::Read<double> Field::get() {
  if (!has()) {
    Omega_h_fail("attempt to read uninitialized "
        "field \"%s\"\n", long_name.c_str());
  }
  return storage;
}

Omega_h::Write<double> Field::set() {
  ensure_allocated();
  return storage;
}

Omega_h::Write<double> Field::getset() {
  if (!has()) {
    Omega_h_fail("attempt to modify uninitialized "
        "field \"%s\"\n", long_name.c_str());
  }
  return storage;
}

void Field::del() {
  storage = decltype(storage)();
}

void Field::finalize_definition(Supports& ss) {
  support = ss.get_support(entity_type, on_points, class_names);
}

void Field::forget_disc() {
  del();
  for (auto& c : conditions) c.forget_disc();
}

void Field::learn_disc() {
  support->subset->learn_disc();
  for (auto& c : conditions) c.learn_disc();
}

bool Field::is_covered_by_conditions(double prev_time, double time) {
  ClassNames covered_class_names;
  for (auto& c : conditions) {
    if (!c.when->active(prev_time, time)) continue;
    auto& names = c.support->subset->class_names;
    covered_class_names.insert(names.begin(), names.end());
  }
  return (covered_class_names == class_names);
}

void Field::apply_conditions(double prev_time, double time,
    Omega_h::Read<double> node_coords, Fields& fields) {
  if (!conditions.empty()) {
    ensure_allocated();
  }
  for (auto& c : conditions) {
    c.apply(prev_time, time, node_coords, fields);
  }
}

double Field::next_event(double time) {
  double out = std::numeric_limits<double>::max();
  for (auto& c : conditions) {
    out = Omega_h::min2(out, c.next_event(time));
  }
  return out;
}

void Field::setup_conditions(Supports& supports, Omega_h::InputList& pl) {
  for (int i = 0; i < pl.size(); ++i) {
    if (pl.is_map(i)) {
      auto& condition_pl = pl.get_map(i);
      conditions.push_back(Condition(this, supports, condition_pl));
    }
  }
}

void Field::setup_default_condition(Supports& supports, double start_time) {
  if (default_value.empty()) return;
  // if users have set some other non-default conditions that fully specify
  // this field at the simulation start time, then don't bother creating
  // a default condition since it'll just be overwritten
  if (this->is_covered_by_conditions(start_time, start_time)) return;
  conditions.insert(conditions.begin(), Condition(this, supports, default_value, support,
        at_time(start_time)));
}

}
