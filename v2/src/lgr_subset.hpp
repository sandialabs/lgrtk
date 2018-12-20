#ifndef LGR_SUBSET_HPP
#define LGR_SUBSET_HPP

#include <Omega_h_array.hpp>
#include <lgr_class_names.hpp>
#include <lgr_element_types.hpp>
#include <lgr_entity_type.hpp>
#include <lgr_field_access.hpp>
#include <lgr_mapping.hpp>

namespace lgr {

struct Disc;

struct Subset {
  Disc& disc;
  EntityType entity_type;
  ClassNames class_names;
  Mapping mapping;
  Subset(Disc& disc_in, EntityType entity_type_in,
      ClassNames const& class_names_in = ClassNames());
  bool is_identity();
  void forget_disc();
  void learn_disc();
  int count();
  MappedElemsToNodes ents_to_nodes();
};

}  // namespace lgr

#endif
