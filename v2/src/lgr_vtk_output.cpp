#include <Omega_h_file.hpp>
#include <Omega_h_profile.hpp>
#include <Omega_h_vtk.hpp>
#include <lgr_field_index.hpp>
#include <lgr_response.hpp>
#include <lgr_simulation.hpp>
#include <lgr_vtk_output.hpp>

#include <Omega_h_print.hpp>

namespace lgr {

using LgrFields = std::vector<FieldIndex>;
using OshFields = std::vector<std::string>;

struct VtkOutput : public Response {
  public:
    VtkOutput(Simulation& sim_in, Omega_h::InputMap& pl);
    void set_fields(Omega_h::InputMap& pl);
    void out_of_line_virtual_method() override final;
    void respond() override final;
  public:
    std::string path;
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
      osh_fields[omega_h_adapt_tags[field_name]].push_back(field_name);
      continue;
    }
    if (omega_h_multi_dim_tags.count(field_name)) {
      if (omega_h_multi_dim_tags[field_name].first == 0) {
        Omega_h::fail(
            "Cannot output field \"%s\" "
            "to VTK for 2nd order meshes\n", field_name.c_str());
      }
      osh_fields[omega_h_multi_dim_tags[field_name].first].push_back(
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
    if (ent_type == ELEMS) {
      if (support->on_points() && (sim.disc.points_per_ent(ELEMS) > 1)) {
        Omega_h_fail(
            "\"%s\" is on integration points, and there is more than one per "
            "element, "
            "and Dan hasn't coded support for visualizing that yet!\n",
            field_name.c_str());
      }
    }
    if (ent_type == NODES) lgr_fields[0].push_back(fi);
    if (ent_type == ELEMS) lgr_fields[stdim].push_back(fi);
  }
}

VtkOutput::VtkOutput(Simulation& sim_in, Omega_h::InputMap& pl) :
  Response(sim_in, pl),
  path(pl.get<std::string>("path", "lgr_viz")),
  pvd_pos(0)
{
  auto comm = sim.comm;
  auto rank = comm->rank();
  if (rank == 0) Omega_h::safe_mkdir(path.c_str());
  comm->barrier();
  auto steps_dir = path + "/steps";
  if (rank == 0) Omega_h::safe_mkdir(steps_dir.c_str());
  comm->barrier();
  if (rank == 0) pvd_pos = Omega_h::vtk::write_initial_pvd(path, sim.time);
  set_fields(pl);
}

static void write_step_dirs(std::string const& step_path,
    Omega_h::CommPtr comm) {
  if (comm->rank() == 0) Omega_h::safe_mkdir(step_path.c_str());
  comm->barrier();
  auto pieces_dir = step_path + "/pieces";
  if (comm->rank() == 0) Omega_h::safe_mkdir(pieces_dir.c_str());
  comm->barrier();
}

static void write_parallel(std::string const& step_path, Simulation& sim,
    LgrFields lgr_fields[4], OshFields osh_fields[4]) {
  OMEGA_H_TIME_FUNCTION;
  write_step_dirs(step_path, sim.comm);
  (void)lgr_fields;
  (void)osh_fields;
}

void VtkOutput::respond() {
  Omega_h::ScopedTimer timer("VtkOutput::respond");
  auto step = sim.step;
  auto time = sim.time;
  auto step_path = path + "/steps/step_" + std::to_string(step);
  write_parallel(step_path, sim, lgr_fields, osh_fields);
  if (this->sim.comm->rank() == 0) {
    Omega_h::vtk::update_pvd(path, &pvd_pos, step, time);
  }
}

void VtkOutput::out_of_line_virtual_method() {
}

Response* vtk_output_factory(
    Simulation& sim, std::string const&, Omega_h::InputMap& pl) {
  return new VtkOutput(sim, pl);
}

}  // namespace lgr
