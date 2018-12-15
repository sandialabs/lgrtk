#ifndef LGR_FIELD_HPP
#define LGR_FIELD_HPP

#include <lgr_condition.hpp>
#include <lgr_entity_type.hpp>
#include <lgr_remap_type.hpp>

namespace lgr {

struct Support;
struct Supports;
struct Fields;

struct Field {
  Field(
      std::string const& short_name_in,
      std::string const& long_name_in,
      int ncomps_in,
      EntityType entity_type_in,
      bool on_points_in,
      ClassNames const& class_names_in,
      bool filling_with_nan_in);
  ~Field() = default;
  std::string short_name;
  std::string long_name;
  int ncomps;
  EntityType entity_type;
  bool on_points;
  ClassNames class_names;
  bool filling_with_nan;
  Support* support;
  Omega_h::Write<double> storage;
  std::string default_value;
  RemapType remap_type;
  std::vector<Condition> conditions;
  bool has();
  void ensure_allocated();
  Omega_h::Read<double> get();
  Omega_h::Write<double> set();
  Omega_h::Write<double> getset();
  void del();
  void finalize_definition(Supports& ss);
  void forget_disc();
  void learn_disc();
  void apply_conditions(double prev_time, double time,
      Omega_h::Read<double> node_coords, Fields& fields);
  bool is_covered_by_conditions(double prev_time, double time);
  double next_event(double time);
  void setup_conditions(Supports& supports, Omega_h::InputList& pl);
  void setup_default_condition(Supports& supports, double start_time);
};

}

#endif
