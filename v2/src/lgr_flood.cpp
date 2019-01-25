#include <Omega_h_array_ops.hpp>
#include <Omega_h_class.hpp>
#include <Omega_h_element.hpp>
#include <Omega_h_file.hpp>
#include <Omega_h_indset_inline.hpp>
#include <Omega_h_map.hpp>
#include <iostream>
#include <lgr_flood.hpp>
#include <lgr_for.hpp>
#include <lgr_simulation.hpp>

namespace lgr {

struct SavedField {
  std::string name;
  Omega_h::Write<double> data;
  Mapping mapping;
};

Flooder::Flooder(Simulation& sim_in) : sim(sim_in) {}

void Flooder::setup(Omega_h::InputMap& pl) {
  enabled = pl.is_map("flood");
  if (!enabled) return;
  auto& flood_pl = pl.get_map("flood");
  max_depth = flood_pl.get<int>("max depth", "1");
  flood_priority = sim.fields.define("flood priority", "flood priority", 1,
      ELEMS, false, sim.disc.covering_class_names());
  sim.fields[flood_priority].remap_type = RemapType::PER_UNIT_VOLUME;
}

void Flooder::flood() {
  if (!enabled) return;
  OMEGA_H_TIME_FUNCTION;
  auto const pull_mapping = choose();
  if (pull_mapping.exists()) {
    flood_by_mapping(pull_mapping);
  }
}

Omega_h::LOs Flooder::choose() {
  Omega_h::ScopedTimer timer("Flooder::choose");
  // step 1: mark candidate vertices (cavity centers)
  auto const qualities = sim.disc.mesh.ask_qualities();
  auto const elems_are_low_qual =
      each_lt(qualities, sim.adapter.opts.min_quality_desired);
  auto verts_are_cands_r =
      mark_down(&sim.disc.mesh, sim.dim(), 0, elems_are_low_qual);
  if (get_max(verts_are_cands_r) != Omega_h::Byte(1)) {
    return Omega_h::LOs();
  }
  auto const verts_to_elems = sim.disc.mesh.ask_graph(0, sim.dim());
  auto const elem_priorities = sim.get(flood_priority);
  auto const nverts = sim.disc.mesh.nverts();
  auto const cavity_source_priorities = Omega_h::Write<double>(nverts);
  auto const cavity_target_priorities = Omega_h::Write<double>(nverts);
  auto const cavity_volumes = Omega_h::Write<double>(nverts);
  auto const point_weights = sim.get(sim.weight);
  auto const points_per_elem = sim.disc.points_per_ent(ELEMS);
  auto const verts_are_cands = deep_copy(verts_are_cands_r);
  // step 2: compute, for each cavity
  // (a) the volume of the smallest material in the cavity
  //     (the one that will get filled)
  // (b) what the lowest priority material in the cavity
  //     is, other than the lowest volume material
  auto predict_functor = OMEGA_H_LAMBDA(int vert) {
    if (!verts_are_cands[vert]) return;
    double min_volume = -1.0;
    double min_volume_priority = -1.0;
    auto const begin = verts_to_elems.a2ab[vert];
    auto const end = verts_to_elems.a2ab[vert + 1];
    for (auto vert_elem = begin; vert_elem < end; ++vert_elem) {
      auto const elem = verts_to_elems.ab2b[vert_elem];
      auto const elem_priority = elem_priorities[elem];
      double priority_volume = 0.0;
      auto const begin2 = verts_to_elems.a2ab[vert];
      auto const end2 = verts_to_elems.a2ab[vert + 1];
      for (auto vert_elem2 = begin2; vert_elem2 < end2; ++vert_elem2) {
        auto const elem2 = verts_to_elems.ab2b[vert_elem2];
        auto const elem2_priority = elem_priorities[elem2];
        if (elem2_priority == elem_priority) {
          double elem_volume = 0.0;
          for (int elem_pt = 0; elem_pt < points_per_elem; ++elem_pt) {
            auto const pt_weight =
                point_weights[elem2 * points_per_elem + elem_pt];
            elem_volume += pt_weight;
          }
          priority_volume += elem_volume;
        }
      }
      if (min_volume == -1.0 || priority_volume < min_volume) {
        min_volume = priority_volume;
        min_volume_priority = elem_priority;
      }
    }
    double min_other_priority = -1.0;
    for (auto vert_elem = begin; vert_elem < end; ++vert_elem) {
      auto const elem = verts_to_elems.ab2b[vert_elem];
      auto const elem_priority = elem_priorities[elem];
      if ((elem_priority != min_volume_priority) &&
          (min_other_priority == -1.0 || elem_priority < min_other_priority)) {
        min_other_priority = elem_priority;
      }
    }
    if (min_other_priority == -1.0) {
      verts_are_cands[vert] = Omega_h::Byte(0);
    } else {
      cavity_source_priorities[vert] = min_other_priority;
      cavity_target_priorities[vert] = min_volume_priority;
      cavity_volumes[vert] = min_volume;
    }
  };
  parallel_for(nverts, std::move(predict_functor));
  verts_are_cands_r = read(verts_are_cands);
  if (get_max(verts_are_cands_r) != Omega_h::Byte(1)) {
    return Omega_h::LOs();
  }
  auto const vert_globals = sim.disc.mesh.globals(0);
  auto compare = OMEGA_H_LAMBDA(int u, int v)->bool {
    auto const u_priority = cavity_source_priorities[u];
    auto const v_priority = cavity_source_priorities[v];
    if (u_priority != v_priority) {
      return u_priority > v_priority;
    }
    auto const u_volume = cavity_volumes[u];
    auto const v_volume = cavity_volumes[v];
    if (u_volume != v_volume) {
      return u_volume > v_volume;
    }
    return vert_globals[u] < vert_globals[v];
  };
  auto const verts_to_verts = sim.disc.mesh.ask_star(0);
  verts_are_cands_r = Omega_h::indset::find(&sim.disc.mesh, 0,
      verts_to_verts.a2ab, verts_to_verts.ab2b, verts_are_cands_r, compare);
  auto const ncavs = get_sum(verts_are_cands_r);
  std::cout << "flooding " << ncavs << " cavities\n";
  auto const nelems = sim.disc.mesh.nelems();
  auto const pull_mapping = Omega_h::Write<int>(nelems, 0, 1);
  OMEGA_H_CHECK(pull_mapping.exists());
  auto apply_functor = OMEGA_H_LAMBDA(int vert) {
    if (!verts_are_cands_r[vert]) return;
    auto const source_priority = cavity_source_priorities[vert];
    int max_volume_source = -1;
    double max_volume = -1.0;
    auto const begin = verts_to_elems.a2ab[vert];
    auto const end = verts_to_elems.a2ab[vert + 1];
    for (auto vert_elem = begin; vert_elem < end; ++vert_elem) {
      auto const elem = verts_to_elems.ab2b[vert_elem];
      auto const elem_priority = elem_priorities[elem];
      if (elem_priority != source_priority) continue;
      double elem_volume = 0.0;
      for (int elem_pt = 0; elem_pt < points_per_elem; ++elem_pt) {
        auto const pt_weight = point_weights[elem * points_per_elem + elem_pt];
        elem_volume += pt_weight;
      }
      if (max_volume_source == -1 || elem_volume > max_volume) {
        max_volume_source = elem;
        max_volume = elem_volume;
      }
    }
    OMEGA_H_CHECK(max_volume_source != -1);
    auto const target_priority = cavity_target_priorities[vert];
    for (auto vert_elem = begin; vert_elem < end; ++vert_elem) {
      auto const elem = verts_to_elems.ab2b[vert_elem];
      auto const elem_priority = elem_priorities[elem];
      if (target_priority == elem_priority) {
        pull_mapping[elem] = max_volume_source;
      }
    }
  };
  parallel_for(nverts, std::move(apply_functor));
  return pull_mapping;
}

void Flooder::flood_by_mapping(Omega_h::LOs pull_mapping) {
  OMEGA_H_TIME_FUNCTION;
  auto const dim = sim.disc.mesh.dim();
  auto const nelems = sim.disc.mesh.nelems();
  auto const elems_will_flood =
      neq_each(pull_mapping, Omega_h::LOs(nelems, 0, 1));
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
      ents_should_declass =
          land_each(ents_should_declass, invert_marks(exposed_sides));
    }
    auto const class_ids_w = deep_copy(old_class_ids[ent_dim]);
    auto const class_dims_w = deep_copy(old_class_dims[ent_dim]);
    auto const clear_functor = OMEGA_H_LAMBDA(int ent) {
      if (ents_should_declass[ent]) {
        class_ids_w[ent] = -1;
        class_dims_w[ent] = Omega_h::I8(dim);
      }
    };
    parallel_for(sim.disc.mesh.nents(ent_dim), std::move(clear_functor));
    sim.disc.mesh.set_tag(ent_dim, "class_id", Omega_h::read(class_ids_w));
    sim.disc.mesh.set_tag(ent_dim, "class_dim", Omega_h::read(class_dims_w));
  }
  sim.disc.mesh.add_tag(dim, "class_id", 1,
      read(unmap(pull_mapping,
          sim.disc.mesh.get_array<Omega_h::ClassId>(dim, "class_id"), 1)));
  Omega_h::finalize_classification(&sim.disc.mesh);
  std::vector<SavedField> saved_fields;
  for (auto const& field_ptr : sim.fields.storage) {
    if (field_ptr->remap_type == RemapType::NONE &&
        field_ptr->long_name != "position")
      continue;
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
  sim.models.learn_disc();
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
    } else
      old_set_size = nelems;
    auto const old_data = saved_field.data;
    OMEGA_H_CHECK(old_data.exists());
    auto const ncomps = divide_no_remainder(old_data.size(), old_set_size);
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
    parallel_for(nelems, std::move(flood_field_functor));
    if (!old_mapping.is_identity) {
      Omega_h::map_value_into(-1, old_mapping.things, old_inverse);
      Omega_h::map_value_into(-1, new_mapping.things, new_inverse);
    }
  }
  std::cout << "done flooding\n";
}

}  // namespace lgr
