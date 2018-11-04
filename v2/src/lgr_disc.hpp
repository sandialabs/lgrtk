#ifndef LGR_DISC_HPP
#define LGR_DISC_HPP

#include <lgr_entity_type.hpp>
#include <lgr_element_types.hpp>
#include <lgr_class_names.hpp>

#include <Omega_h_mesh.hpp>
#include <Omega_h_input.hpp>

namespace lgr {

struct Disc {
  int dim();
  int count(EntityType type);
  void setup(Omega_h::CommPtr comm, Omega_h::InputMap& pl);
  Omega_h::LOs ents_to_nodes(EntityType type);
  Omega_h::Adj nodes_to_ents(EntityType type);
  Omega_h::LOs ents_on_closure(
      ClassNames const& class_names,
      EntityType type);
  ClassNames const& covering_class_names();
  int nodes_per_ent(EntityType type);
  int points_per_ent(EntityType type);
  template <class Elem>
  void set_elem();
  Omega_h::Reals node_coords();
  Omega_h::Mesh mesh;
  int dim_;
  bool is_simplex_;
  int points_per_ent_[4];
  int nodes_per_ent_[4];
  ClassNames covering_class_names_;
  Omega_h::LOs elems2nodes_;
  Omega_h::Adj nodes2elems_;
};

#define LGR_EXPL_INST(Elem) \
extern template void Disc::set_elem<Elem>();
LGR_EXPL_INST_ELEMS
#undef LGR_EXPL_INST

}

#endif
