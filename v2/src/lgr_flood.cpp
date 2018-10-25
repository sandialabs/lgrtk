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

void Flooder::flood() {
  if (!enabled) return;
  std::cout << "flooding...\n";
  auto const status = schedule();
  if (!status.some_were_bad) {
    std::cout << "done flooding (all fixed)\n";
    return;
  }
  if (status.some_did_flood) {
    flood_once(status.pull_mapping, status.elems_did_flood);
    std::cout << "done flooding (did flood)\n";
    return;
  }
  std::cout << "done flooding (out of options)\n";
}

Flooder::FloodStatus Flooder::schedule() {
  OMEGA_H_TIME_FUNCTION;
  FloodStatus status;
  auto const dim = sim.disc.mesh.dim();
  auto const nelems = sim.disc.mesh.nelems();
  auto const qualities = sim.disc.mesh.ask_qualities();
  auto const elems_floodable_priority = Omega_h::Write<double>(nelems);
  auto const min_quality_desired = sim.adapter.opts.min_quality_desired;
  auto const elems_to_priority = sim.get(flood_priority);
  auto init_floodable = OMEGA_H_LAMBDA(int elem) {
    if (qualities[elem] < min_quality_desired) {
      elems_floodable_priority[elem] = elems_to_priority[elem];
    } else {
      elems_floodable_priority[elem] = -1.0;
    }
  };
  parallel_for(nelems, std::move(init_floodable));
  auto const elems_can_flood = each_neq_to(read(elems_floodable_priority), -1.0);
  auto nfloodable = get_sum(elems_can_flood);
  std::cout << nfloodable << " bad elements\n";
  status.some_were_bad = (get_max(elems_can_flood) == Omega_h::Byte(1));
  if (!status.some_were_bad) return status;
  for (int i = 0; i < max_depth; ++i) {
    auto const nverts = sim.disc.mesh.nverts();
    auto const verts_floodable_priority = Omega_h::Write<double>(nverts);
    auto const verts2elems = sim.disc.mesh.ask_graph(0, sim.dim());
    auto const floodable_down = OMEGA_H_LAMBDA(int vert) {
      double vert_priority = -1.0;
      for (auto vert_elem = verts2elems.a2ab[vert];
          vert_elem < verts2elems.a2ab[vert + 1]; ++vert_elem) {
        auto const elem = verts2elems.ab2b[vert_elem];
        auto const elem_priority = elems_floodable_priority[elem];
        if (elem_priority != -1.0) {
          if (vert_priority == -1.0 || elem_priority < vert_priority) {
            vert_priority = elem_priority;
          }
        }
      }
      verts_floodable_priority[vert] = vert_priority;
    };
    parallel_for(nverts, std::move(floodable_down));
    auto const verts_per_elem = Omega_h::element_degree(sim.disc.mesh.family(), sim.dim(), 0);
    auto const elems2verts = sim.disc.mesh.ask_elem_verts();
    auto const floodable_up = OMEGA_H_LAMBDA(int elem) {
      double elem_priority = -1.0;
      for (int elem_vert = 0; elem_vert < verts_per_elem; ++elem_vert) {
        auto const vert = elems2verts[elem * verts_per_elem + elem_vert];
        auto const vert_priority = verts_floodable_priority[vert];
        if (vert_priority != -1.0) {
          if (elem_priority == -1.0 || vert_priority < elem_priority) {
            elem_priority = vert_priority;
          }
        }
      }
      elems_floodable_priority[elem] = elem_priority;
    };
    parallel_for(nelems, std::move(floodable_up));
  }
  auto const side_class_dims = sim.disc.mesh.get_array<Omega_h::I8>(dim - 1, "class_dim");
  Omega_h::Write<int> pull_mapping(nelems, 0, 1);
  OMEGA_H_CHECK(sim.disc.points_per_ent(ELEMS) == 1);
  auto const elems_did_flood = Omega_h::Write<Omega_h::Byte>(nelems);
  auto const sides_per_elem = Omega_h::element_degree(sim.disc.mesh.family(), dim, dim - 1);
  auto const elems_to_sides = sim.disc.mesh.ask_down(dim, dim - 1).ab2b;
  auto const sides_to_elems = sim.disc.mesh.ask_up(dim - 1, dim);
  auto pull_decision_functor = OMEGA_H_LAMBDA(int elem) {
    auto best_elem = elem;
    auto const self_priority = elems_to_priority[best_elem];
    if (elems_floodable_priority[elem] == self_priority) {
      auto best_priority = -1.0;
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
          if (other_priority != self_priority && (best_priority == -1.0 || other_priority < best_priority)) {
            best_elem = other_elem;
            best_priority = other_priority;
          }
        }
      }
    }
    elems_did_flood[elem] = (best_elem != elem) ? Omega_h::Byte(1) : Omega_h::Byte(0);
    pull_mapping[elem] = best_elem;
  };
  parallel_for("flood pull decision", nelems, std::move(pull_decision_functor));
  status.some_did_flood = (get_max(read(elems_did_flood)) == Omega_h::Byte(1));
  status.pull_mapping = pull_mapping;
  status.elems_did_flood = elems_did_flood;
  return status;
}

// returns true iff a deeper flooding should be called
void Flooder::flood_once(Omega_h::LOs pull_mapping, Omega_h::Bytes elems_did_flood) {
  OMEGA_H_TIME_FUNCTION;
  std::cout << "flood_once" << '\n';
  auto const dim = sim.disc.mesh.dim();
  auto const nelems = sim.disc.mesh.nelems();
  Omega_h::Few<Omega_h::Read<Omega_h::I8>, 3> old_class_dims;
  Omega_h::Few<Omega_h::Read<Omega_h::ClassId>, 3> old_class_ids;
  for (int ent_dim = 0; ent_dim < dim; ++ent_dim) {
    old_class_dims[ent_dim] =
      sim.disc.mesh.get_array<Omega_h::I8>(ent_dim, "class_dim");
    old_class_ids[ent_dim] =
      sim.disc.mesh.get_array<Omega_h::ClassId>(
          ent_dim, "class_id");
    auto ents_should_declass =
      mark_down(&sim.disc.mesh, dim, ent_dim, elems_did_flood);
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
