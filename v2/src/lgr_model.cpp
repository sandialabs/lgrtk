#include <lgr_model.hpp>
#include <lgr_simulation.hpp>
#include <lgr_element_types.hpp>
#include <lgr_support.hpp>
#include <lgr_subset.hpp>

namespace lgr {

void ModelBase::out_of_line_virtual_function() {}

ModelBase::ModelBase(Simulation& sim_in, ClassNames const& class_names)
:sim(sim_in)
{
  elem_support = sim.supports.get_support(ELEMS, false, class_names);
  point_support = sim.supports.get_support(ELEMS, true, class_names);
}

static ClassNames class_names_from_pl(Simulation& sim, Teuchos::ParameterList& pl) {
  auto class_names_teuchos = pl.get<Teuchos::Array<std::string>>("sets", Teuchos::Array<std::string>());
  ClassNames class_names(class_names_teuchos.begin(), class_names_teuchos.end());
  if (class_names.empty()) class_names = sim.disc.covering_class_names();
  return class_names;
}

ModelBase::ModelBase(Simulation& sim_in, Teuchos::ParameterList& pl):
  ModelBase(sim_in, class_names_from_pl(sim_in, pl))
{
}

FieldIndex ModelBase::elem_define(
    std::string const& short_name, std::string const& long_name,
    int ncomps) {
  return sim.fields.define(short_name, long_name, ncomps, elem_support);
}

FieldIndex ModelBase::elem_define(
    std::string const& short_name, std::string const& long_name,
    int ncomps, std::string const& default_value) {
  auto fi = elem_define(short_name, long_name, ncomps);
  sim.fields[fi].default_value = default_value;
  return fi;
}

FieldIndex ModelBase::elem_define(
    std::string const& short_name, std::string const& long_name,
    int ncomps, RemapType tt, std::string const& default_value) {
  auto fi = elem_define(short_name, long_name, ncomps);
  sim.fields[fi].default_value = default_value;
  sim.fields[fi].remap_type = tt;
  return fi;
}

FieldIndex ModelBase::point_define(
    std::string const& short_name, std::string const& long_name,
    int ncomps) {
  return sim.fields.define(short_name, long_name, ncomps, point_support);
}

FieldIndex ModelBase::point_define(
    std::string const& short_name, std::string const& long_name,
    int ncomps, std::string const& default_value) {
  auto fi = point_define(short_name, long_name, ncomps);
  sim.fields[fi].default_value = default_value;
  return fi;
}

FieldIndex ModelBase::point_define(
    std::string const& short_name, std::string const& long_name,
    int ncomps, RemapType tt, std::string const& default_value) {
  auto fi = point_define(short_name, long_name, ncomps);
  sim.fields[fi].default_value = default_value;
  sim.fields[fi].remap_type = tt;
  return fi;
}

MappedElemsToNodes ModelBase::get_elems_to_nodes() {
  MappedElemsToNodes out;
  out.mapping = elem_support->subset->mapping;
  out.data = sim.disc.ents_to_nodes(ELEMS);
  return out;
}

int ModelBase::points() {
  return point_support->count();
}

MappedRead ModelBase::elems_get(FieldIndex fi) {
  return sim.get(fi, elem_support->subset);
}

MappedWrite ModelBase::elems_set(FieldIndex fi) {
  return sim.set(fi, elem_support->subset);
}

MappedWrite ModelBase::elems_getset(FieldIndex fi) {
  return sim.getset(fi, elem_support->subset);
}

template <class Elem>
Model<Elem>::Model(Simulation& sim_in, Teuchos::ParameterList& pl)
:ModelBase(sim_in, pl)
{
}

template <class Elem>
Model<Elem>::Model(Simulation& sim_in, ClassNames const& class_names)
:ModelBase(sim_in, class_names)
{
}

template <class Elem>
MappedPointRead<Elem> Model<Elem>::points_get(FieldIndex fi) {
  return sim.points_get<Elem>(fi, point_support->subset);
}

template <class Elem>
MappedPointWrite<Elem> Model<Elem>::points_set(FieldIndex fi) {
  return sim.points_set<Elem>(fi, point_support->subset);
}

template <class Elem>
MappedPointWrite<Elem> Model<Elem>::points_getset(FieldIndex fi) {
  return sim.points_getset<Elem>(fi, point_support->subset);
}

#define LGR_EXPL_INST(Elem) \
template struct Model<Elem>;
LGR_EXPL_INST_ELEMS
#undef LGR_EXPL_INST

}
