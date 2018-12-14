#ifndef LGR_SIMULATION_HPP
#define LGR_SIMULATION_HPP

#include <lgr_disc.hpp>
#include <lgr_subsets.hpp>
#include <lgr_supports.hpp>
#include <lgr_fields.hpp>
#include <lgr_factories.hpp>
#include <lgr_input_variables.hpp>
#include <lgr_models.hpp>
#include <lgr_field_access.hpp>
#include <lgr_element_types.hpp>
#include <lgr_scalars.hpp>
#include <lgr_responses.hpp>
#include <lgr_adapt.hpp>
#include <lgr_flood.hpp>
#include <Omega_h_timer.hpp>

namespace lgr {

struct Simulation {
  std::string elem_name;
  Omega_h::CommPtr comm;
  Factories factories;
  InputVariables input_variables;
  Disc disc;
  Subsets subsets;
  Supports supports;
  Fields fields;
  Models models;
  Scalars scalars;
  Responses responses;
  Adapter adapter;
  Flooder flooder;
  Simulation(Omega_h::CommPtr comm, Factories&& factories_in);
  double get_double(Omega_h::InputMap& pl, const char* name, const char* default_expr);
  int get_int(Omega_h::InputMap& pl, const char* name, const char* default_expr);
  template <class Elem>
  void set_elem();
  void setup(Omega_h::InputMap& pl);
  int dim();
  int nodes();
  int elems();
  int points();
  Omega_h::LOs elems_to_nodes();
  Omega_h::Adj nodes_to_elems();
  void finalize_definitions();
  bool has(FieldIndex fi);
  Omega_h::Read<double> get(FieldIndex fi);
  Omega_h::Write<double> set(FieldIndex fi);
  Omega_h::Write<double> getset(FieldIndex fi);
  MappedRead get(FieldIndex fi, Subset* subset);
  MappedWrite set(FieldIndex fi, Subset* subset);
  MappedWrite getset(FieldIndex fi, Subset* subset);
  template <class Elem>
  MappedPointRead<Elem> points_get(FieldIndex fi, Subset* subset);
  template <class Elem>
  MappedPointWrite<Elem> points_set(FieldIndex fi, Subset* subset);
  template <class Elem>
  MappedPointWrite<Elem> points_getset(FieldIndex fi, Subset* subset);
  void del(FieldIndex fi);
  double time;
  double end_time;
  double prev_time;
  double dt;
  double prev_dt;
  double max_dt;
  int step;
  int end_step;
  double cfl;
  FieldIndex position;
  FieldIndex velocity;
  FieldIndex acceleration;
  FieldIndex force;
  FieldIndex stress;
  FieldIndex gradient;
  FieldIndex weight;
  FieldIndex time_step_length;
  FieldIndex viscosity_length;
  FieldIndex wave_speed;
  FieldIndex density;
  FieldIndex lumping;
  FieldIndex nodal_mass;
  FieldIndex point_time_step;
  FieldIndex traction;
  FieldIndex traction_weight;
  Omega_h::Now start_cpu_time_point;
  double prev_cpu_time;
  double cpu_time;
  double min_dt;
};

void apply_conditions(Simulation& sim,
    FieldIndex fi);
void apply_conditions(Simulation& sim);

void update_time(Simulation& sim);
void update_cpu_time(Simulation& sim);

#define LGR_EXPL_INST(Elem) \
extern template void Simulation::set_elem<Elem>();
LGR_EXPL_INST_ELEMS
#undef LGR_EXPL_INST

#define LGR_EXPL_INST(Elem) \
extern template MappedPointRead<Elem> Simulation::points_get<Elem>(FieldIndex, Subset*); \
extern template MappedPointWrite<Elem> Simulation::points_set<Elem>(FieldIndex, Subset*); \
extern template MappedPointWrite<Elem> Simulation::points_getset<Elem>(FieldIndex, Subset*);
LGR_EXPL_INST_ELEMS_AND_SIDES
#undef LGR_EXPL_INST

}

#endif
