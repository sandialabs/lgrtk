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

bool flood(Simulation& sim) {
  if (!sim.enable_flooding) return false;
  OMEGA_H_TIME_FUNCTION;
  std::cerr << "flooding!\n";
  auto const qualities = sim.disc.mesh.ask_qualities();
  auto const desired_quality = sim.adapter.opts.min_quality_desired;
  auto const elems_could_flood = Omega_h::each_lt(qualities, desired_quality);
  auto const floodable_elems = Omega_h::collect_marked(elems_could_flood);
  if (floodable_elems.size() == 0) return false;
  std::cerr << floodable_elems.size() << " floodable elems!\n";
  Omega_h::Write<int> elems_to_pull_from(floodable_elems.size(), -1);
  auto const densities = sim.get(sim.density);
  auto const dim = sim.disc.mesh.dim();
  auto const elems_to_sides = sim.disc.mesh.ask_down(dim, dim - 1).ab2b;
  auto const sides_to_elems = sim.disc.mesh.ask_up(dim - 1, dim);
  auto const sides_per_elem = Omega_h::element_degree(sim.disc.mesh.family(), dim, dim - 1);
  auto const side_class_dims = sim.disc.mesh.get_array<Omega_h::I8>(dim - 1, "class_dim");
  OMEGA_H_CHECK(sim.disc.points_per_ent(ELEMS) == 1);
  auto decide_functor = OMEGA_H_LAMBDA(int floodable_elem) {
    auto const elem = floodable_elems[floodable_elem];
    std::cerr << "elem " << elem << "is floodable!\n";
    auto const density = densities[elem]; // single point specific !!!
    int best_floodable = -1;
    double best_floodable_density = Omega_h::ArithTraits<double>::max();
    int best_non_floodable = -1;
    double best_non_floodable_density = Omega_h::ArithTraits<double>::max();
    for (int elem_side = 0; elem_side < sides_per_elem; ++elem_side) {
      auto const side = elems_to_sides[elem * sides_per_elem + elem_side];
      // don't flood from the same material
      if (int(side_class_dims[side]) != dim - 1) continue;
      for (auto side_elem = sides_to_elems.a2ab[side];
          side_elem < sides_to_elems.a2ab[side + 1]; ++side_elem) {
        auto const other_elem = sides_to_elems.ab2b[side_elem];
        if (other_elem == elem) continue;
        auto const other_density = densities[other_elem];
        if (elems_could_flood[other_elem]) {
          if ((other_density < density) &&
              (other_density < best_floodable_density)) {
            best_floodable = other_elem;
            best_floodable_density = other_density;
          }
        } else {
          if (other_density < best_non_floodable_density) {
            best_non_floodable = other_elem;
            best_non_floodable_density = other_density;
          }
        }
      }
    }
    int best = -1;
    if (best_non_floodable != -1) {
      std::cerr << "elem " << elem << " is flooding from best non-floodable " << best_non_floodable << '\n';
      best = best_non_floodable;
    } else if (best_floodable != -1) {
      std::cerr << "elem " << elem << " is flooding from best floodable " << best_floodable << '\n';
      best = best_floodable;
    } else {
      std::cerr << "elem " << elem << " isn't flooding!\n";
      best = elem;
    }
    elems_to_pull_from[floodable_elem] = best;
  };
  parallel_for("flood decision", floodable_elems.size(), std::move(decide_functor));
  auto const old_elem_class_ids = sim.disc.mesh.get_array<Omega_h::ClassId>(dim, "class_id");
  auto const elem_class_ids = Omega_h::deep_copy(old_elem_class_ids);
  auto const flood_class_functor = OMEGA_H_LAMBDA(int floodable_elem) {
    auto const pull_from = elems_to_pull_from[floodable_elem];
    auto const elem = floodable_elems[floodable_elem];
    if (pull_from != elem) {
      elem_class_ids[elem] = elem_class_ids[pull_from];
    }
  };
  parallel_for("flood class", floodable_elems.size(), std::move(flood_class_functor));
  auto const elems_did_flood = Omega_h::neq_each(old_elem_class_ids, Omega_h::read(elem_class_ids));
  if (Omega_h::get_min(elems_did_flood) == Omega_h::Byte(0)) {
    sim.disc.mesh.add_tag(dim, "old class_id", 1, elems_did_flood);
  }
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
  auto const nelems = sim.disc.mesh.nelems();
  Omega_h::Write<int> pull_mapping(nelems, 0, 1);
  Omega_h::Write<int> old_inverse(nelems, -1);
  Omega_h::Write<int> new_inverse(nelems, -1);
  Omega_h::map_into(
      Omega_h::read(elems_to_pull_from), floodable_elems, pull_mapping, 1);
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
    auto flood_field_functor = OMEGA_H_LAMBDA(int elem_to) {
      auto const elem_from = pull_mapping[elem_to];
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
        new_data[new_set_elem * ncomps + comp] =
          old_data[old_set_elem * ncomps + comp];
      }
    };
    parallel_for("flood field", nelems,
        std::move(flood_field_functor));
    if (!old_mapping.is_identity) {
      Omega_h::map_value_into(-1, old_mapping.things, old_inverse);
      Omega_h::map_value_into(-1, new_mapping.things, new_inverse);
    }
  }
  return true;
}

}
