#include <lgr_flood.hpp>
#include <lgr_simulation.hpp>
#include <lgr_for.hpp>
#include <Omega_h_file.hpp>
#include <Omega_h_array_ops.hpp>
#include <Omega_h_map.hpp>
#include <Omega_h_element.hpp>
#include <Omega_h_class.hpp>

namespace lgr {

struct SavedField {
  std::string name;
  Omega_h::Write<double> data;
  Mapping mapping;
};

Flooder::Flooder(Simulation& sim_in):
  sim(sim_in)
{
}

void Flooder::setup(Teuchos::ParameterList& pl)
{
  enabled = pl.isSublist("flood");
  if (!enabled) return;
  auto& flood_pl = pl.sublist("flood");
  max_depth = flood_pl.get<int>("max depth", 3);
  flood_priority = sim.fields.define("flood priority",
      "flood priority", 1, ELEMS, false,
      sim.disc.covering_class_names());
  sim.fields[flood_priority].remap_type =
    RemapType::PER_UNIT_VOLUME;
}

void Flooder::flood() {
  auto const max_priority = get_max(sim.get(flood_priority));
  for (int depth = 0; depth < max_depth; ++depth) {
    for (double up_to_priority = 1.0; up_to_priority <= max_priority;
        up_to_priority += 1.0) {
      auto const status = flood_once(depth, up_to_priority);
      if (!status.some_were_bad) return;
      if (status.some_did_flood) return;
    }
  }
}

// returns true iff a deeper flooding should be called
Flooder::FloodStatus Flooder::flood_once(int depth, double up_to_priority) {
  OMEGA_H_TIME_FUNCTION;
  FloodStatus status;
  auto const dim = sim.disc.mesh.dim();
  auto const old_elem_class_ids = sim.disc.mesh.get_array<Omega_h::ClassId>(dim, "class_id");
  auto const nelems = sim.disc.mesh.nelems();
  auto const qualities = sim.disc.mesh.ask_qualities();
  auto elems_can_flood = each_lt(qualities,
      sim.adapter.opts.min_quality_desired);
  status.some_were_bad = (get_max(elems_can_flood) == Omega_h::Byte(1));
  if (!status.some_were_bad) return status;
  for (int i = 0; i < depth; ++i) {
    auto const adj_verts = mark_down(
        &sim.disc.mesh, sim.dim(), 0, elems_can_flood);
    elems_can_flood = mark_up(
        &sim.disc.mesh, 0, sim.dim(), adj_verts);
  }
  auto const floodable_elems = collect_marked(elems_can_flood);
  auto const elems_to_priority = sim.get(flood_priority);
  auto const side_class_dims = sim.disc.mesh.get_array<Omega_h::I8>(dim - 1, "class_dim");
  auto const elem_class_ids = Omega_h::Write<Omega_h::ClassId>(nelems);
  Omega_h::Write<int> pull_mapping(nelems, 0, 1);
  OMEGA_H_CHECK(sim.disc.points_per_ent(ELEMS) == 1);
  auto const elems_did_flood = Omega_h::Write<Omega_h::Byte>(nelems);
  auto const sides_per_elem = Omega_h::element_degree(sim.disc.mesh.family(), dim, dim - 1);
  auto const elems_to_sides = sim.disc.mesh.ask_down(dim, dim - 1).ab2b;
  auto const sides_to_elems = sim.disc.mesh.ask_up(dim - 1, dim);
  auto pull_decision_functor = OMEGA_H_LAMBDA(int elem) {
    auto best_elem = elem;
    if (elems_can_flood[elem]) {
      auto const self_priority = elems_to_priority[best_elem];
      if (self_priority <= up_to_priority) {
        auto best_priority = self_priority;
        for (int elem_side = 0; elem_side < sides_per_elem;
            ++elem_side) {
          auto const side =
            elems_to_sides[elem * sides_per_elem + elem_side];
          for (auto side_elem = sides_to_elems.a2ab[side];
              side_elem < sides_to_elems.a2ab[side + 1];
              ++side_elem) {
            auto const other_elem = sides_to_elems.ab2b[side_elem];
            if (other_elem == elem) continue;
            auto const other_priority = 
              elems_to_priority[other_elem];
            if (other_priority < best_priority) {
              best_elem = other_elem;
              best_priority = other_priority;
            }
          }
        }
      }
    }
    elems_did_flood[elem] = (best_elem != elem) ? Omega_h::Byte(1) : Omega_h::Byte(0);
    pull_mapping[elem] = best_elem;
    elem_class_ids[elem] = old_elem_class_ids[best_elem];
  };
  parallel_for("flood pull decision", nelems, std::move(pull_decision_functor));
  status.some_did_flood = (get_max(read(elems_did_flood)) == Omega_h::Byte(1));
  if (!status.some_did_flood) return status;
  Omega_h::Few<Omega_h::Read<Omega_h::I8>, 3> old_class_dims;
  Omega_h::Few<Omega_h::Read<Omega_h::ClassId>, 3> old_class_ids;
  for (int ent_dim = 0; ent_dim < dim; ++ent_dim) {
    old_class_dims[ent_dim] =
      sim.disc.mesh.get_array<Omega_h::I8>(ent_dim, "class_dim");
    old_class_ids[ent_dim] =
      sim.disc.mesh.get_array<Omega_h::ClassId>(
          ent_dim, "class_id");
    auto const ents_in_flood_closure =
      mark_down(&sim.disc.mesh, dim, ent_dim, elems_did_flood);
    auto const class_ids_w = deep_copy(old_class_ids[ent_dim]);
    auto const class_dims_w = deep_copy(old_class_dims[ent_dim]);
    auto const clear_functor = OMEGA_H_LAMBDA(int ent) {
      if (ents_in_flood_closure[ent]) {
        class_ids_w[ent] = -1;
        class_dims_w[ent] = Omega_h::I8(dim);
      }
    };
    parallel_for("clear flooded lowers",
        sim.disc.mesh.nents(ent_dim), std::move(clear_functor));
    sim.disc.mesh.set_tag(ent_dim, "class_id", Omega_h::read(class_ids_w));
    sim.disc.mesh.set_tag(ent_dim, "class_dim", Omega_h::read(class_dims_w));
  }
  sim.disc.mesh.add_tag(dim, "class_id", 1, Omega_h::read(elem_class_ids));
  Omega_h::finalize_classification(&sim.disc.mesh);
  std::vector<SavedField> saved_fields;
  for (auto const& field_ptr : sim.fields.storage) {
    if (field_ptr->remap_type == RemapType::NONE && field_ptr->long_name != "position") continue;
    SavedField saved_field;
    saved_field.name = field_ptr->long_name;
    saved_field.data = field_ptr->storage;
    saved_field.mapping = field_ptr->support->subset->mapping;
    saved_fields.push_back(std::move(saved_field));
  }
  sim.fields.forget_disc();
  sim.subsets.forget_disc();
  sim.subsets.learn_disc();
  sim.fields.learn_disc();
  Omega_h::Write<int> old_inverse(nelems, -1);
  Omega_h::Write<int> new_inverse(nelems, -1);
  for (auto& saved_field : saved_fields) {
    auto const fi = sim.fields.find(saved_field.name);
    auto& field = sim.fields[fi];
    auto const old_mapping = saved_field.mapping;
    auto const new_mapping = field.support->subset->mapping;
    if (field.entity_type != ELEMS || field.remap_type == RemapType::SHAPE) {
      field.storage = saved_field.data;
      OMEGA_H_CHECK((old_mapping.is_identity && new_mapping.is_identity) ||
          (old_mapping.things.size() == new_mapping.things.size()));
      continue;
    }
    int old_set_size;
    if (!old_mapping.is_identity) {
      Omega_h::inject_map(old_mapping.things, old_inverse);
      Omega_h::inject_map(new_mapping.things, new_inverse);
      old_set_size = old_mapping.things.size();
    } else old_set_size = nelems;
    auto const old_data = saved_field.data;
    OMEGA_H_CHECK(old_data.exists());
    auto const ncomps = divide_no_remainder(
        old_data.size(), old_set_size);
    auto const new_data = sim.set(fi);
    OMEGA_H_CHECK(new_data.exists());
    auto const debug_c_str = field.long_name.c_str();
    auto flood_field_functor = OMEGA_H_LAMBDA(int elem_to) {
      auto const elem_from = pull_mapping[elem_to];
      OMEGA_H_CHECK(elem_from != -1);
      int old_set_elem, new_set_elem;
      if (old_mapping.is_identity) {
        old_set_elem = elem_from;
        new_set_elem = elem_to;
      } else {
        old_set_elem = old_inverse[elem_from];
        if (old_set_elem == -1) return;
        new_set_elem = new_inverse[elem_to];
      }
      for (int comp = 0; comp < ncomps; ++comp) {
        if ((!(new_set_elem * ncomps + comp < new_data.size())) || (new_set_elem * ncomps + comp < 0)) {
          Omega_h_fail("field \"%s\" new_set_elem %d ncomps %d comp %d new_data.size() %d\n",
              debug_c_str, new_set_elem, ncomps, comp, new_data.size());
        }
        if ((!(old_set_elem * ncomps + comp < old_data.size())) || (old_set_elem * ncomps + comp < 0)) {
          Omega_h_fail("field \"%s\" old_set_elem %d ncomps %d comp %d old_data.size() %d\n",
              debug_c_str, old_set_elem, ncomps, comp, old_data.size());
        }
        new_data[new_set_elem * ncomps + comp] =
          old_data[old_set_elem * ncomps + comp];
      }
    };
    parallel_for("flood field", nelems, std::move(flood_field_functor));
    if (!old_mapping.is_identity) {
      Omega_h::map_value_into(-1, old_mapping.things, old_inverse);
      Omega_h::map_value_into(-1, new_mapping.things, new_inverse);
    }
  }
  return status;
}

}
