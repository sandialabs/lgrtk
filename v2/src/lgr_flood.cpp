#include <lgr_flood.hpp>
#include <lgr_simulation.hpp>
#include <lgr_for.hpp>
#include <Omega_h_file.hpp>
#include <Omega_h_array_ops.hpp>
#include <Omega_h_map.hpp>
#include <Omega_h_element.hpp>
#include <Omega_h_class.hpp>
#include <iostream>

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

void Flooder::setup(Omega_h::InputMap& pl)
{
  enabled = pl.is_map("flood");
  if (!enabled) return;
  auto& flood_pl = pl.get_map("flood");
  max_depth = flood_pl.get<int>("max depth", "1");
  flood_priority = sim.fields.define("flood priority",
      "flood priority", 1, ELEMS, false,
      sim.disc.covering_class_names());
  sim.fields[flood_priority].remap_type =
    RemapType::PER_UNIT_VOLUME;
}

Omega_h::LOs Flooder::choose() {
  OMEGA_H_TIME_FUNCTION;
  OMEGA_H_CHECK(sim.disc.mesh.comm()->size() == 1);
  // here we go...
  // this is all based on the principle of "destroy the loneliest material"
  // harsh as it may sound, this is at least a provably convergent method
  // (if we repeatedly remove the smallest material volume in a cavity, eventually
  //  the cavity will be single-material)
  // Step 1: identify the low-quality elements
  auto const qualities = sim.disc.mesh.ask_qualities();
  auto const elems_are_low_qual = each_lt(qualities, sim.adapter.opts.min_quality_desired);
  // Step 2: decide who each vertex is working for, based on worst quality
  auto const verts2elems = sim.disc.mesh.ask_graph(0, sim.dim());
  auto const nverts = sim.disc.mesh.nverts();
  auto const vert_loyalties = Omega_h::Write<int>(nverts);
  auto vert_loyalty_functor = OMEGA_H_LAMBDA(int vert) {
    int loyal_to = -1;
    double lowest_quality;
    for (auto vert_elem = verts2elems.a2ab[vert]; vert_elem < verts2elems.a2ab[vert + 1]; ++vert_elem) {
      auto const elem = verts2elems.ab2b[vert_elem];
      auto const elem_qual = qualities[elem];
      if (!elems_are_low_qual[elem]) continue;
      if (loyal_to == -1 || elem_qual < lowest_quality) {
        loyal_to = elem;
        lowest_quality = elem_qual;
      }
    }
    vert_loyalties[vert] = loyal_to;
  };
  parallel_for(nverts, std::move(vert_loyalty_functor));
  // Step 3: for each vertex working for a low-quality element, identify its smallest surrounding
  // material volume, which we'll call the target material (identified by a priority)
  auto const vert_target_priorities = Omega_h::Write<double>(nverts);
  auto const vert_target_volumes = Omega_h::Write<double>(nverts);
  auto const points_to_weights = sim.get(sim.weight);
  auto const points_per_elem = sim.disc.points_per_ent(ELEMS);
  auto const elem_priorities = sim.get(flood_priority);
  auto vert_target_functor = OMEGA_H_LAMBDA(int vert) {
    if (vert_loyalties[vert] == -1) return;
    double target_priority = -1.0;
    double lowest_volume;
    for (auto vert_elem = verts2elems.a2ab[vert]; vert_elem < verts2elems.a2ab[vert + 1]; ++vert_elem) {
      auto const elem = verts2elems.ab2b[vert_elem];
      auto const elem_priority = elem_priorities[elem];
      auto const priority_volume = 0.0;
      for (auto vert_elem2 = verts2elems.a2ab[vert]; vert_elem2 < verts2elems.a2ab[vert + 1]; ++vert_elem2) {
        auto const elem2 = verts2elems.ab2b[vert_elem2];
        auto const elem2_priority = elem_priorities[elem2];
        if (elem2_priority != elem_priority) continue;
        for (int point = 0; point < points_per_elem; ++point) {
          priority_volume += points_to_weights[elem2 * points_per_elem + point];
        }
      }
      if (target_priority == -1.0 || priority_volume < lowest_volume) {
        target_priority = elem_priority;
        lowest_volume = priority_volume;
      }
    }
    vert_target_priorities[vert] = target_priority;
    vert_target_volumes[vert] = lowest_volume;
  };
  parallel_for(nverts, std::move(vert_target_functor));
  // Step 4: for each low-quality element, all of whose vertices are working for it, choose the lowest-volume
  // material amongst the per-vertex targets to designate as the overall target
  auto const nelems = sim.disc.mesh.nelems();
  auto const elem_has_full_loyalty = Omega_h::Write<Omega_h::Byte>(nelems);
  auto const verts_per_elem = Omega_h::element_degree(sim.disc.mesh.family(), sim.dim(), 0);
  auto const elems2verts = sim.disc.mesh.ask_elem_verts();
  auto const elem_target_priorities = Omega_h::Write<double>(nelems);
  auto const elem_target_volumes = Omega_h::Write<double>(nelems);
  auto elem_declare_functor = OMEGA_H_LAMBDA(int elem) {
    if (!elems_are_low_qual[elem]) return;
    double target_priority = -1.0;
    double lowest_volume;
    for (int elem_vert = 0; elem_vert < verts_per_elem; ++elem_vert) {
      auto const vert = elems2verts[elem * verts_per_elem + elem_vert];
      if (vert_loyalties[vert] != elem) {
        elem_has_full_loyalty[elem] = Omega_h::Byte(0);
        return;
      }
      auto const vert_volume = vert_target_volumes[vert];
      if (target_priority == -1.0 || vert_volume < lowest_volume) {
        target_priority = vert_target_priorities[vert];
        lowest_volume = vert_volume;
      }
    }
    elem_has_full_loyalty[elem] = Omega_h::Byte(0);
    elem_target_priorities[elem] = target_priority;
    elem_target_volumes[elem] = lowest_volume;
  };
  parallel_for(nelems, std::move(elem_declare_functor));
  // Step 5: for each vertex, if it is loyal to an element that commands full loyalty,
  // then record the quality of that element and set the vertex's target priority to that of the element
  // otherwise, remove its loyalty
  auto const vert_loyal_qualities = Omega_h::Write<double>(nverts);
  auto vert_loyalty_pivot = OMEGA_H_LAMBDA(int vert) {
    auto const loyal_to = vert_loyalties[vert];
    if (loyal_to == -1) return;
    if (!elem_has_full_loyalty[loyal_to]) {
      vert_loyalties[vert] = -1;
      return;
    }
    vert_loyal_qualities[vert] = qualities[loyal_to];
    vert_target_priorities[vert] = elem_target_priorities[loyal_to];
  };
  parallel_for(nverts, std::move(vert_loyalty_pivot));
  // Step 6: for each element, find its adjacent vertex which is loyal to the lowest quality element (if any).
  // if found, and if the element's priority is the vertex's target priority, mark said element for flooding
  auto const elems_can_flood = Omega_h::Write<Omega_h::Byte>(nelems);
  auto elem_flood_allow = OMEGA_H_LAMBDA(int elem) {
    double target_priority = -1.0;
    double lowest_quality;
    for (int elem_vert = 0; elem_vert < verts_per_elem; ++elem_vert) {
      auto const vert = elems2verts[elem * verts_per_elem + elem_vert];
      if (vert_loyalties[vert] == -1) continue;
      auto const vert_priority = vert_target_priorities[vert];
      auto const vert_quality = vert_loyal_qualities[vert];
      if (target_priority == -1.0 || vert_quality < lowest_quality) {
        target_priority = vert_priority;
        lowest_quality = vert_quality;
      }
    }
    elems_can_flood[elem] = (target_priority == elem_priorities[elem]) ? Omega_h::Byte(1) : Omega_h::Byte(0);
  };
  parallel_for(nelems, std::move(elem_flood_allow));
  // Step 7: for each element allowed to flood, find a good nearby candidate to pull material from
}

void Flooder::flood_once(Omega_h::LOs pull_mapping) {
  OMEGA_H_TIME_FUNCTION;
  std::cout << "flood_once" << '\n';
  auto const dim = sim.disc.mesh.dim();
  auto const nelems = sim.disc.mesh.nelems();
  auto const elems_will_flood = neq_each(pull_mapping, Omega_h::LOs(nelems, 0, 1));
  Omega_h::Few<Omega_h::Read<Omega_h::I8>, 3> old_class_dims;
  Omega_h::Few<Omega_h::Read<Omega_h::ClassId>, 3> old_class_ids;
  for (int ent_dim = 0; ent_dim < dim; ++ent_dim) {
    old_class_dims[ent_dim] =
      sim.disc.mesh.get_array<Omega_h::I8>(ent_dim, "class_dim");
    old_class_ids[ent_dim] =
      sim.disc.mesh.get_array<Omega_h::ClassId>(ent_dim, "class_id");
    auto ents_should_declass =
      mark_down(&sim.disc.mesh, dim, ent_dim, elems_will_flood);
    // mae sure not declassify domain boundary sides
    if (ent_dim == dim - 1) {
      auto const exposed_sides = mark_exposed_sides(&sim.disc.mesh);
      ents_should_declass = land_each(ents_should_declass,
          invert_marks(exposed_sides));
    }
    auto const class_ids_w = deep_copy(old_class_ids[ent_dim]);
    auto const class_dims_w = deep_copy(old_class_dims[ent_dim]);
    auto const clear_functor = OMEGA_H_LAMBDA(int ent) {
      if (ents_should_declass[ent]) {
        class_ids_w[ent] = -1;
        class_dims_w[ent] = Omega_h::I8(dim);
      }
    };
    parallel_for("clear flooded lowers",
        sim.disc.mesh.nents(ent_dim), std::move(clear_functor));
    sim.disc.mesh.set_tag(ent_dim, "class_id", Omega_h::read(class_ids_w));
    sim.disc.mesh.set_tag(ent_dim, "class_dim", Omega_h::read(class_dims_w));
  }
  sim.disc.mesh.add_tag(dim, "class_id", 1, read(unmap(pull_mapping, sim.disc.mesh.get_array<Omega_h::ClassId>(dim, "class_id"), 1)));
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
}

}
