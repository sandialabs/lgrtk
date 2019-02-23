#ifndef LGR_FIELDS_HPP
#define LGR_FIELDS_HPP

#include <lgr_field.hpp>
#include <lgr_field_index.hpp>
#include <memory>

namespace lgr {

struct Fields {
  std::vector<std::unique_ptr<Field>> storage;
  bool printing_set_fields;
  bool filling_with_nan;
  std::vector<FieldIndex> set_fields;
  void setup(Omega_h::InputMap& pl);
  FieldIndex define(std::string const& short_name, std::string const& long_name,
      int ncomps, EntityType type, bool on_points,
      ClassNames const& class_names);
  FieldIndex define(std::string const& short_name, std::string const& long_name,
      int ncomps, Support* support);
  void finalize_definitions(Supports& s);
  Field& operator[](FieldIndex fi);
  bool is_allocated(FieldIndex fi);
  Omega_h::Read<double> get(FieldIndex fi);
  Omega_h::Write<double> getset(FieldIndex fi);
  Omega_h::Write<double> set(FieldIndex fi);
  void del(FieldIndex fi);
  double next_event(double time);
  void setup_conditions(Simulation& sim, Omega_h::InputMap& pl);
  void setup_common_defaults(Omega_h::InputMap& pl);
  FieldIndex find(std::string const& name);
  void print_and_clear_set_fields();
  void setup_default_conditions(Simulation& sim, double start_time);
  void forget_disc();
  void learn_disc();
  void copy_to_omega_h(Disc& disc, std::vector<FieldIndex> field_indices);
  void copy_from_omega_h(Disc& disc, std::vector<FieldIndex> field_indices);
  void remove_from_omega_h(Disc& disc, std::vector<FieldIndex> field_indices);
  bool has(std::string const& name);
};

}  // namespace lgr

#endif
