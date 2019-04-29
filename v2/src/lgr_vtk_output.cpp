#include <Omega_h_file.hpp>
#include <Omega_h_for.hpp>
#include <Omega_h_map.hpp>
#include <Omega_h_profile.hpp>
#include <Omega_h_vtk.hpp>
#include <fstream>
#include <lgr_field_index.hpp>
#include <lgr_response.hpp>
#include <lgr_simulation.hpp>
#include <lgr_vtk_output.hpp>

namespace lgr {

struct VtkOutput : public Response {
 public:
  VtkOutput(Simulation& sim_in, Omega_h::InputMap& pl);
  void set_fields(Omega_h::InputMap& pl);
  void out_of_line_virtual_method() override final;
  void respond() override final;

 public:
  bool compress;
  Omega_h::filesystem::path path;
  std::streampos pvd_pos;
  LgrFields lgr_fields[4];
  OshFields osh_fields[4];
};

void VtkOutput::set_fields(Omega_h::InputMap& pl) {
  auto stdim = static_cast<std::size_t>(sim.dim());
  std::map<std::string, std::size_t> omega_h_adapt_tags = {
      {"quality", stdim}, {"metric", 0}};
  std::map<std::string, std::pair<std::size_t, std::string>>
      omega_h_multi_dim_tags = {{"element class_id", {stdim, "class_id"}},
          {"node class_dim", {0, "class_dim"}}, {"node local", {0, "local"}},
          {"node global", {0, "global"}}, {"element local", {stdim, "local"}},
          {"element global", {stdim, "global"}}};
  auto& field_names_in = pl.get_list("fields");
  for (int i = 0; i < field_names_in.size(); ++i) {
    auto field_name = field_names_in.get<std::string>(i);
    if (omega_h_adapt_tags.count(field_name)) {
      if (!sim.disc.mesh.has_tag(0, "metric")) {
        if (sim.disc.is_second_order_) {
          Omega_h::fail(
              "Cannot output field \"metric\""
              "to VTK for 2nd order meshes\n");
        }
        Omega_h::add_implied_isos_tag(&sim.disc.mesh);
      }
      osh_fields[omega_h_adapt_tags[field_name]].insert(field_name);
      continue;
    }
    if (omega_h_multi_dim_tags.count(field_name)) {
      if (omega_h_multi_dim_tags[field_name].first == 0 &&
          sim.disc.is_second_order_) {
        Omega_h::fail(
            "Cannot output field \"%s\" "
            "to VTK for 2nd order meshes\n",
            field_name.c_str());
      }
      osh_fields[omega_h_multi_dim_tags[field_name].first].insert(
          omega_h_multi_dim_tags[field_name].second);
      continue;
    }
    auto fi = sim.fields.find(field_name);
    if (!fi.is_valid()) {
      Omega_h_fail(
          "Cannot visualize "
          "undefined field \"%s\"\n",
          field_name.c_str());
    }
    auto support = sim.fields[fi].support;
    auto ent_type = support->subset->entity_type;
    if (!(ent_type == NODES || ent_type == ELEMS)) {
      Omega_h_fail(
          "\"%s\" is not on nodes or elements, VTK can't visualize it!\n",
          field_name.c_str());
    }
    if (ent_type == NODES) lgr_fields[0].insert(fi.storage_index);
    if (ent_type == ELEMS) lgr_fields[stdim].insert(fi.storage_index);
  }
}

VtkOutput::VtkOutput(Simulation& sim_in, Omega_h::InputMap& pl)
    : Response(sim_in, pl),
      compress(false),
      path(pl.get<std::string>("path", "lgr_viz")),
      pvd_pos(0) {
  set_fields(pl);
  if (sim.no_output) return;
  auto const comm = sim.comm;
  auto const rank = comm->rank();
  if (rank == 0) Omega_h::filesystem::create_directory(path);
  comm->barrier();
  auto const steps_dir = path / "steps";
  if (rank == 0) Omega_h::filesystem::create_directory(steps_dir);
  comm->barrier();
  if (rank == 0) pvd_pos = Omega_h::vtk::write_initial_pvd(path, sim.time);
}

static void write_step_dirs(
    Omega_h::filesystem::path const& step_path, Omega_h::CommPtr comm) {
  if (comm->rank() == 0) Omega_h::filesystem::create_directory(step_path);
  comm->barrier();
  auto const pieces_dir = step_path / "pieces";
  if (comm->rank() == 0) Omega_h::filesystem::create_directory(pieces_dir);
  comm->barrier();
}

static void describe_osh_tags(
    std::ostream& file, Simulation& sim, OshFields& osh_fields, int ent_dim) {
  auto mesh = sim.disc.mesh;
  for (int i = 0; i < mesh.ntags(ent_dim); ++i) {
    auto tag = mesh.get_tag(ent_dim, i);
    if (osh_fields.count(tag->name())) {
      Omega_h::vtk::write_p_tag(file, tag, mesh.dim());
    }
  }
}

static void describe_multi_point_lgr_field(
    std::ostream& file, Simulation& sim, Field& field, int ent_dim) {
  OMEGA_H_CHECK(ent_dim == sim.disc.dim());
  auto npoints = sim.disc.points_per_ent(ELEMS);
  for (int pt = 0; pt < npoints; ++pt) {
    auto pt_name = field.long_name + "_" + std::to_string(pt);
    Omega_h::vtk::write_p_data_array<double>(file, pt_name, field.ncomps);
  }
}

static void describe_lgr_fields(
    std::ostream& file, Simulation& sim, LgrFields lgr_fields, int ent_dim) {
  for (auto it : lgr_fields) {
    FieldIndex fi;
    fi.storage_index = it;
    auto& field = sim.fields[fi];
    auto support = field.support;
    if (support->on_points() && (sim.disc.points_per_ent(ELEMS) > 1)) {
      describe_multi_point_lgr_field(file, sim, field, ent_dim);
    } else {
      Omega_h::vtk::write_p_data_array<double>(
          file, field.long_name, field.ncomps);
    }
  }
}

static Omega_h::filesystem::path piece_filename(int rank) {
  Omega_h::filesystem::path result("pieces");
  result /= "piece_";
  result += std::to_string(rank);
  result += ".vtu";
  return result;
}

static void write_pvtu(Omega_h::filesystem::path const& step_path,
    Simulation& sim, LgrFields lgr_fields[4], OshFields osh_fields[4]) {
  if (sim.comm->rank() != 0) return;
  auto const dim = sim.disc.mesh.dim();
  auto const pvtu_name = step_path / "pieces.pvtu";
  std::ofstream file(pvtu_name.c_str());
  OMEGA_H_CHECK(file.is_open());
  file << "<VTKFile type=\"PUnstructuredGrid\">\n";
  file << "<PUnstructuredGrid>\n";
  file << "<PPoints>\n";
  Omega_h::vtk::write_p_data_array<Omega_h::Real>(file, "coordinates", 3);
  file << "</PPoints>\n";
  file << "<PPointData>\n";
  describe_osh_tags(file, sim, osh_fields[0], 0);
  describe_lgr_fields(file, sim, lgr_fields[0], 0);
  file << "</PPointData>\n";
  file << "<PCellData>\n";
  describe_osh_tags(file, sim, osh_fields[dim], dim);
  describe_lgr_fields(file, sim, lgr_fields[dim], dim);
  file << "</PCellData>\n";
  for (int i = 0; i < sim.comm->size(); ++i) {
    file << "<Piece Source=\"" << piece_filename(i) << "\"/>\n";
  }
  file << "</PUnstructuredGrid>\n";
  file << "</VTKFile>\n";
}

static void write_piece_start_tag(std::ostream& file, Simulation& sim) {
  file << "<Piece NumberOfPoints=\"" << sim.disc.count(NODES) << "\"";
  file << " NumberOfCells=\"" << sim.disc.count(ELEMS) << "\">\n";
}

enum {
  VTK_VERTEX = 1,
  VTK_LINE = 3,
  VTK_TRIANGLE = 5,
  VTK_QUAD = 9,
  VTK_TETRA = 10,
  VTK_HEXAHEDRON = 12,
  VTK_QUADRATIC_EDGE = 21,
  VTK_QUADRATIC_TRIANGLE = 22,
  VTK_QUADRATIC_QUAD = 23,
  VTK_QUADRATIC_TETRA = 24,
  VTK_QUADRATIC_HEXAHEDRON = 25
};

static Omega_h::I8 vtk_type(Disc& disc) {
  auto dim = disc.dim();
  auto family = disc.mesh.family();
  int orderm1 = (disc.is_second_order_ == true) ? 1 : 0;
  if (family == OMEGA_H_SIMPLEX) {
    Omega_h::I8 simplex_table[4][2] = {{VTK_VERTEX, VTK_VERTEX},
        {VTK_LINE, VTK_QUADRATIC_EDGE}, {VTK_TRIANGLE, VTK_QUADRATIC_TRIANGLE},
        {VTK_TETRA, VTK_QUADRATIC_TETRA}};
    return simplex_table[dim][orderm1];
  } else if (family == OMEGA_H_HYPERCUBE) {
    Omega_h::I8 hypercube_table[4][2] = {{VTK_VERTEX, VTK_VERTEX},
        {VTK_LINE, VTK_QUADRATIC_EDGE}, {VTK_QUAD, VTK_QUADRATIC_QUAD},
        {VTK_HEXAHEDRON, VTK_QUADRATIC_HEXAHEDRON}};
    return hypercube_table[dim][orderm1];
  }
  return 0;
}

static void write_connectivity(
    std::ostream& file, Simulation& sim, bool compress) {
  auto disc = sim.disc;
  Omega_h::Read<Omega_h::I8> types(disc.count(ELEMS), vtk_type(disc));
  Omega_h::vtk::write_array(file, "types", 1, types, compress);
  auto elems2nodes = disc.ents_to_nodes(ELEMS);
  auto deg = disc.nodes_per_ent(ELEMS);
  Omega_h::LOs ends(disc.count(ELEMS), deg, deg);
  Omega_h::vtk::write_array(file, "connectivity", 1, elems2nodes, compress);
  Omega_h::vtk::write_array(file, "offsets", 1, ends, compress);
}

static void write_coords(std::ostream& file, Simulation& sim, bool compress) {
  auto dim = sim.disc.dim();
  auto coords = sim.disc.get_node_coords();
  auto coords3 = Omega_h::resize_vectors(coords, dim, 3);
  Omega_h::vtk::write_array(file, "coordinates", 3, coords3, compress);
}

static void write_osh_tags(std::ostream& file, Simulation& sim,
    OshFields osh_fields, int ent_dim, bool compress) {
  auto mesh = sim.disc.mesh;
  for (int i = 0; i < mesh.ntags(ent_dim); ++i) {
    auto tag = mesh.get_tag(ent_dim, i);
    if (osh_fields.count(tag->name())) {
      Omega_h::vtk::write_tag(file, tag, mesh.dim(), compress);
    }
  }
}

template <typename T>
Omega_h::Read<T> gather_pt(
    Omega_h::Read<T> data, int nents, int npoints, int ncomps, int pt) {
  Omega_h::Write<T> pt_data(nents * ncomps);
  auto functor = OMEGA_H_LAMBDA(int ent) {
    for (int c = 0; c < ncomps; ++c) {
      pt_data[ent * ncomps + c] = data[(ent * npoints + pt) * ncomps + c];
    }
  };
  Omega_h::parallel_for("gather point", nents, std::move(functor));
  return pt_data;
}

static void write_subset_array(std::ostream& file, std::string const& name,
    int const ncomps, Omega_h::Reals subset_data, bool compress,
    Mapping const& mapping, int const nents) {
  if (mapping.is_identity) {
    Omega_h::vtk::write_array(file, name, ncomps, subset_data, compress);
  } else {
    auto full_data =
        Omega_h::map_onto(subset_data, mapping.things, nents, 0.0, ncomps);
    Omega_h::vtk::write_array(file, name, ncomps, full_data, compress);
  }
}

static void write_multi_point_lgr_field(
    std::ostream& file, Simulation& sim, Field& field, bool compress) {
  auto data = field.get();
  auto ncomps = field.ncomps;
  auto nents = sim.disc.count(ELEMS);
  auto npoints = sim.disc.points_per_ent(ELEMS);
  for (int pt = 0; pt < npoints; ++pt) {
    auto pt_data = gather_pt(data, nents, npoints, ncomps, pt);
    auto pt_name = field.long_name + "_" + std::to_string(pt);
    write_subset_array(file, pt_name, ncomps, pt_data, compress,
        field.support->subset->mapping, nents);
  }
}

static void write_lgr_fields(std::ostream& file, Simulation& sim,
    LgrFields lgr_fields, int ent_dim, bool compress) {
  for (auto it : lgr_fields) {
    FieldIndex fi;
    fi.storage_index = it;
    auto& field = sim.fields[fi];
    auto support = field.support;
    if (support->on_points() && (sim.disc.points_per_ent(ELEMS) > 1)) {
      write_multi_point_lgr_field(file, sim, field, compress);
    } else {
      auto const array = field.get();
      auto const ent_type = (ent_dim == 0) ? NODES : ELEMS;
      auto const nents = sim.disc.count(ent_type);
      write_subset_array(file, field.long_name, field.ncomps, field.get(),
          compress, field.support->subset->mapping, nents);
    }
  }
}

void write_vtu(Omega_h::filesystem::path const& step_path,
    Simulation& sim, bool compress, LgrFields lgr_fields[4],
    OshFields osh_fields[4], bool override_path) {
  OMEGA_H_TIME_FUNCTION;
  auto const dim = sim.disc.dim();
  Omega_h::filesystem::path vtu_name;
  if (override_path) {
    sim.disc.set_node_coords(sim.get(sim.position));
    vtu_name = step_path;
  }
  else vtu_name = step_path / piece_filename(sim.comm->rank());
  std::ofstream file(vtu_name.c_str());
  Omega_h::vtk::write_vtkfile_vtu_start_tag(file, compress);
  file << "<UnstructuredGrid>\n";
  write_piece_start_tag(file, sim);
  file << "<Cells>\n";
  write_connectivity(file, sim, compress);
  file << "</Cells>\n";
  file << "<Points>\n";
  write_coords(file, sim, compress);
  file << "</Points>\n";
  file << "<PointData>\n";
  write_osh_tags(file, sim, osh_fields[0], 0, compress);
  write_lgr_fields(file, sim, lgr_fields[0], 0, compress);
  file << "</PointData>\n";
  file << "<CellData>\n";
  write_osh_tags(file, sim, osh_fields[dim], dim, compress);
  write_lgr_fields(file, sim, lgr_fields[dim], dim, compress);
  file << "</CellData>\n";
  file << "</Piece>\n";
  file << "</UnstructuredGrid>\n";
  file << "</VTKFile>\n";
}

static void write_parallel(Omega_h::filesystem::path const& step_path,
    Simulation& sim, bool compress, LgrFields lgr_fields[4],
    OshFields osh_fields[4]) {
  OMEGA_H_TIME_FUNCTION;
  write_step_dirs(step_path, sim.comm);
  write_pvtu(step_path, sim, lgr_fields, osh_fields);
  write_vtu(step_path, sim, compress, lgr_fields, osh_fields);
}

void VtkOutput::respond() {
  Omega_h::ScopedTimer timer("VtkOutput::respond");
  if (sim.no_output) return;
  sim.disc.set_node_coords(sim.get(sim.position));
  auto const step = sim.step;
  auto const time = sim.time;
  auto step_path = path;
  step_path /= "steps";
  step_path /= "step_";
  step_path += std::to_string(step);
  write_parallel(step_path, sim, compress, lgr_fields, osh_fields);
  if (this->sim.comm->rank() == 0) {
    Omega_h::vtk::update_pvd(path, &pvd_pos, step, time);
  }
}

void VtkOutput::out_of_line_virtual_method() {}

Response* vtk_output_factory(
    Simulation& sim, std::string const&, Omega_h::InputMap& pl) {
  return new VtkOutput(sim, pl);
}

}  // namespace lgr
