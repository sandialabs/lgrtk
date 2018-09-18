#include <lgr_subset.hpp>
#include <lgr_disc.hpp>
#include <Omega_h_map.hpp>
#include <lgr_field_access.hpp>
#include <lgr_for.hpp>

namespace lgr {

Subset::Subset(
    Disc& disc_in,
    EntityType entity_type_in,
    ClassNames const& class_names_in):
  disc(disc_in),
  entity_type(entity_type_in),
  class_names(class_names_in)
{
  mapping.is_identity = (class_names == disc.covering_class_names());
  learn_disc();
}

bool Subset::is_identity() { return mapping.is_identity; }

void Subset::forget_disc() {
  mapping.things = decltype(mapping.things)();
}

void Subset::learn_disc() {
  if (mapping.is_identity) return;
  if (!mapping.things.exists()) {
    mapping.things = disc.ents_on_closure(
        class_names, entity_type);
  }
}

int Subset::count() {
  if (mapping.is_identity) return disc.count(entity_type);
  return mapping.things.size();
}

MappedElemsToNodes Subset::ents_to_nodes() {
  MappedElemsToNodes out;
  out.mapping = mapping;
  out.data = disc.ents_to_nodes(entity_type);
  return out;
}

}
