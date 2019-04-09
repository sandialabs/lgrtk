#include <Omega_h_map.hpp>
#include <Omega_h_profile.hpp>
#include <algorithm>
#include <lgr_disc.hpp>
#include <lgr_fields.hpp>
#include <lgr_subset.hpp>
#include <lgr_subsets.hpp>
#include <lgr_support.hpp>
#include <lgr_for.hpp>

// DEBUG!
#include <iostream>

namespace lgr {

void Fields::setup(Omega_h::InputMap& pl) {
  printing_set_fields = pl.get<bool>("print all fields", "false");
  filling_with_nan = pl.get<bool>("initialize with NaN", "false");
}

FieldIndex Fields::define(std::string const& short_name,
    std::string const& long_name, int ncomps, EntityType type, bool on_points,
    ClassNames const& class_names) {
  auto it = std::find_if(
      storage.begin(), storage.end(), [&](std::unique_ptr<Field> const& f) {
        return f->long_name == long_name;
      });
  if (it == storage.end()) {
    auto ptr = new Field(short_name, long_name, ncomps, type, on_points,
        class_names, filling_with_nan);
    std::unique_ptr<Field> uptr(ptr);
    it = storage.insert(it, std::move(uptr));
  } else {
    auto ptr = it->get();
    OMEGA_H_CHECK(ptr->short_name == short_name);
    OMEGA_H_CHECK(ptr->long_name == long_name);
    OMEGA_H_CHECK(ptr->ncomps == ncomps);
    OMEGA_H_CHECK(ptr->entity_type == type);
    OMEGA_H_CHECK(ptr->on_points == on_points);
    ptr->class_names.insert(class_names.begin(), class_names.end());
  }
  FieldIndex fi;
  fi.storage_index = decltype(fi.storage_index)(it - storage.begin());
  return fi;
}

FieldIndex Fields::define(std::string const& short_name,
    std::string const& long_name, int ncomps, Support* support) {
  return define(short_name, long_name, ncomps, support->subset->entity_type,
      support->on_points(), support->subset->class_names);
}

void Fields::finalize_definitions(Supports& ss) {
  for (auto& field : storage) {
    field->finalize_definition(ss);
  }
}

static void check_index(FieldIndex fi) {
  if (!fi.is_valid()) {
    Omega_h_fail("attempt to use invalid field index\n");
  }
}

bool Fields::has(FieldIndex fi) {
  check_index(fi);
  return storage[fi.storage_index]->has();
}

Field& Fields::operator[](FieldIndex fi) {
  check_index(fi);
  return *storage[fi.storage_index];
}

Omega_h::Read<double> Fields::get(FieldIndex fi) {
  check_index(fi);
  return storage[fi.storage_index]->get();
}

Omega_h::Write<double> Fields::getset(FieldIndex fi) {
  check_index(fi);
  if (printing_set_fields) set_fields.push_back(fi);
  return storage[fi.storage_index]->getset();
}

Omega_h::Write<double> Fields::set(FieldIndex fi) {
  check_index(fi);
  if (printing_set_fields) set_fields.push_back(fi);
  return storage[fi.storage_index]->set();
}

void Fields::del(FieldIndex fi) {
  check_index(fi);
  storage[fi.storage_index]->del();
}

double Fields::next_event(double time) {
  double out = std::numeric_limits<double>::max();
  for (auto& field : storage) {
    out = Omega_h::min2(out, field->next_event(time));
  }
  return out;
}

void Fields::setup_default_conditions(Simulation& sim, double start_time) {
  for (auto& field : storage) {
    field->setup_default_condition(sim, start_time);
  }
}

void Fields::setup_conditions(Simulation& sim, Omega_h::InputMap& pl) {
  for (auto& field_name : pl) {
    if (pl.is_list(field_name)) {
      auto& field_pl = pl.get_list(field_name);
      auto fit = std::find_if(
          storage.begin(), storage.end(), [&](std::unique_ptr<Field> const& f) {
            return f->long_name == field_name;
          });
      if (fit == storage.end()) {
        Omega_h_fail(
            "trying to apply conditions to field \"%s\" that wasn't defined\n",
            field_name.c_str());
      }
      auto& field = *fit;
      field->setup_conditions(sim, field_pl);
    }
  }
}

void Fields::setup_common_defaults(Omega_h::InputMap& pl) {
  for (auto& field_name : pl) {
    if (pl.is<std::string>(field_name)) {
      auto value = pl.get<std::string>(field_name);
      auto fit = std::find_if(
          storage.begin(), storage.end(), [&](std::unique_ptr<Field> const& f) {
            return f->long_name == field_name;
          });
      if (fit == storage.end()) {
        Omega_h_fail(
            "trying to apply default condition to field \"%s\" that wasn't "
            "defined\n",
            field_name.c_str());
      }
      auto& field = *fit;
      field->default_value = value;
    }
  }
}

FieldIndex Fields::find(std::string const& name) {
  FieldIndex out;
  auto it = std::find_if(storage.begin(), storage.end(),
      [&](std::unique_ptr<Field> const& f) { return f->long_name == name; });
  if (it == storage.end()) {
    Omega_h_fail("Could not find field \"%s\"\n", name.c_str());
  }
  out.storage_index = decltype(out.storage_index)(it - storage.begin());
  return out;
}

void Fields::print_and_clear_set_fields() {
  if (!printing_set_fields) return;
  for (auto fi : set_fields) {
    auto& f = operator[](fi);
    auto r = Omega_h::read(f.storage);
    auto hr = Omega_h::HostRead<double>(r);
    std::printf("%s: {", f.long_name.c_str());
    for (int i = 0; i < hr.size(); ++i) {
      if (i) std::printf(", ");
      std::printf("%f", hr[i]);
    }
    std::printf("}\n");
  }
  set_fields.clear();
}

void Fields::forget_disc() {
  Omega_h::ScopedTimer timer("Fields::forget_disc");
  for (auto& field : storage) {
    field->forget_disc();
  }
}

void Fields::learn_disc() {
  Omega_h::ScopedTimer timer("Fields::forget_disc");
  for (auto& field : storage) {
    field->learn_disc();
  }
}

static void copy_field_on_all_nodes_to_omega_h(Omega_h::Mesh& mesh, std::string const& name, Omega_h::Reals const nodal_data, int num_nodes,
  bool is_second_order,
  Omega_h::Few<Omega_h::LOs, 2> p2_nodes)
{
  auto const ncomps = divide_no_remainder(nodal_data.size(), num_nodes);
  if (!is_second_order) {
    mesh.add_tag(0, name, ncomps, nodal_data);
    return;
  }
  auto const vertex_nodes = p2_nodes[0];
  Omega_h::Write<double> vertex_data(vertex_nodes.size() * ncomps);
  auto vertex_functor = OMEGA_H_LAMBDA(int const vertex) {
    auto const node = vertex_nodes[vertex];
    for (int comp = 0; comp < ncomps; ++comp) {
      vertex_data[vertex * ncomps + comp] = nodal_data[node * ncomps + comp];
    }
  };
  Omega_h::parallel_for(mesh.nverts(), std::move(vertex_functor));
  auto const edge_nodes = p2_nodes[1];
  OMEGA_H_CHECK(edge_nodes.size() == mesh.nedges());
  Omega_h::Write<double> edge_data(edge_nodes.size() * ncomps);
  auto edge_functor = OMEGA_H_LAMBDA(int const edge) {
    auto const node = edge_nodes[edge];
    for (int comp = 0; comp < ncomps; ++comp) {
      double const val = nodal_data[node * ncomps + comp];
      edge_data[edge * ncomps + comp] = val;
    }
  };
  Omega_h::parallel_for(mesh.nedges(), std::move(edge_functor));
  mesh.add_tag(0, name, ncomps, read(vertex_data));
  mesh.add_tag(1, name, ncomps, read(edge_data));
}

static void copy_field_on_all_nodes_from_omega_h(Omega_h::Mesh& mesh, std::string const& name, Omega_h::Write<double>& nodal_data_out, int num_nodes,
  bool is_second_order,
  Omega_h::Few<Omega_h::LOs, 2> p2_nodes)
{
  if (!is_second_order) {
    nodal_data_out = deep_copy(mesh.get_array<double>(0, name));
    return;
  }
  auto const vertex_nodes = p2_nodes[0];
  auto const vertex_data = mesh.get_array<double>(0, name);
  auto const ncomps = divide_no_remainder(vertex_data.size(), mesh.nverts());
  Omega_h::Write<double> nodal_data(num_nodes * ncomps);
  auto vertex_functor = OMEGA_H_LAMBDA(int const vertex) {
    auto const node = vertex_nodes[vertex];
    for (int comp = 0; comp < ncomps; ++comp) {
      nodal_data[node * ncomps + comp] = vertex_data[vertex * ncomps + comp];
    }
  };
  Omega_h::parallel_for(mesh.nverts(), std::move(vertex_functor));
  auto const edge_data = mesh.get_array<double>(1, name);
  auto const edge_nodes = p2_nodes[1];
  auto edge_functor = OMEGA_H_LAMBDA(int const edge) {
    auto const node = edge_nodes[edge];
    for (int comp = 0; comp < ncomps; ++comp) {
      nodal_data[node * ncomps + comp] = edge_data[edge * ncomps + comp];
    }
  };
  Omega_h::parallel_for(mesh.nedges(), std::move(edge_functor));
  nodal_data_out = nodal_data;
}

static void copy_field_on_some_nodes_to_omega_h(Omega_h::Mesh& mesh, Field& field, int num_nodes,
  bool is_second_order,
  Omega_h::Few<Omega_h::LOs, 2> p2_nodes)
{
  auto data = field.storage;
  auto& mapping = field.support->subset->mapping;
  if (!field.support->subset->mapping.is_identity) {
    auto ncomps = divide_no_remainder(data.size(), mapping.things.size());
    data = Omega_h::unmap(mapping.things, read(data), ncomps);
  }
  copy_field_on_all_nodes_to_omega_h(mesh, field.long_name, data, num_nodes, is_second_order, p2_nodes);
}

void Fields::copy_field_to_mesh_coordinates(Disc& disc, Field& field) {
  copy_field_on_all_nodes_to_omega_h(disc.mesh, "coordinates", field.storage, disc.count(NODES),
      disc.is_second_order_, disc.p2_nodes);
}

void Fields::copy_field_from_mesh_coordinates(Disc& disc, Field& field) {
  field.set();
  copy_field_on_all_nodes_from_omega_h(disc.mesh, "coordinates", field.storage, disc.count(NODES),
      disc.is_second_order_, disc.p2_nodes);
}

static void copy_field_on_some_nodes_from_omega_h(Omega_h::Mesh& mesh, Field& field, int num_nodes,
  bool is_second_order,
  Omega_h::Few<Omega_h::LOs, 2> p2_nodes)
{
  Omega_h::Write<double> data;
  copy_field_on_all_nodes_from_omega_h(mesh, field.long_name, data, num_nodes, is_second_order, p2_nodes);
  auto& mapping = field.support->subset->mapping;
  if (!field.support->subset->mapping.is_identity) {
    auto ncomps = divide_no_remainder(data.size(), mapping.things.size());
    data = Omega_h::unmap(mapping.things, read(data), ncomps);
  }
  field.storage = data;
}

static void remove_field_on_nodes_from_omega_h(Omega_h::Mesh& mesh, Field& field,
  bool is_second_order)
{
  mesh.remove_tag(0, field.long_name);
  if (is_second_order) mesh.remove_tag(1, field.long_name);
}

void Fields::copy_to_omega_h(
    Disc& disc, std::vector<FieldIndex> field_indices) {
  for (auto fi : field_indices) {
    auto& field = operator[](fi);
    if (field.entity_type != ELEMS) continue;
    auto& mapping = field.support->subset->mapping;
    auto const data = field.get();
    if (field.support->subset->mapping.is_identity) {
      auto ncomps =
          divide_no_remainder(data.size(), disc.mesh.nents(disc.dim()));
      disc.mesh.add_tag(disc.dim(), field.long_name, ncomps, data);
    } else {
      auto ncomps = divide_no_remainder(data.size(), mapping.things.size());
      auto full_data = Omega_h::map_onto(
          data, mapping.things, disc.mesh.nents(disc.dim()), 0.0, ncomps);
      disc.mesh.add_tag(disc.dim(), field.long_name, ncomps, full_data);
    }
  }
  for (auto fi : field_indices) {
    auto& field = operator[](fi);
    if (field.entity_type != NODES) continue;
    copy_field_on_some_nodes_to_omega_h(disc.mesh, field, disc.count(NODES), disc.is_second_order_, disc.p2_nodes);
  }
}

void Fields::copy_from_omega_h(
    Disc& disc, std::vector<FieldIndex> field_indices) {
  for (auto fi : field_indices) {
    auto& field = operator[](fi);
    if (field.entity_type != ELEMS) continue;
    auto& mapping = field.support->subset->mapping;
    if (field.support->subset->mapping.is_identity) {
      field.storage = Omega_h::deep_copy(
          disc.mesh.get_array<double>(disc.dim(), field.long_name),
          field.long_name);
    } else {
      auto tag = disc.mesh.get_tag<double>(disc.dim(), field.long_name);
      auto full_data = tag->array();
      auto ncomps = tag->ncomps();
      auto subset_data = Omega_h::unmap(mapping.things, full_data, ncomps);
      field.storage = subset_data;
    }
  }
  for (auto fi : field_indices) {
    auto& field = operator[](fi);
    if (field.entity_type != NODES) continue;
    std::cout << "copying " << field.long_name << " from Omega_h\n";
    copy_field_on_some_nodes_from_omega_h(disc.mesh, field, disc.count(NODES), disc.is_second_order_, disc.p2_nodes);
  }
}

void Fields::remove_from_omega_h(
    Disc& disc, std::vector<FieldIndex> field_indices) {
  for (auto fi : field_indices) {
    auto& field = operator[](fi);
    if (field.entity_type != ELEMS) continue;
    disc.mesh.remove_tag(disc.dim(), field.long_name);
  }
  for (auto fi : field_indices) {
    auto& field = operator[](fi);
    if (field.entity_type != NODES) continue;
    remove_field_on_nodes_from_omega_h(disc.mesh, field, disc.is_second_order_);
  }
}

}  // namespace lgr
