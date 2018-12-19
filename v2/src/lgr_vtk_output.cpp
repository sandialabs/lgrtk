#include <Omega_h_file.hpp>
#include <Omega_h_profile.hpp>
#include <Omega_h_vtk.hpp>
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
    std::string path;
    std::streampos pvd_pos;
    std::vector<FieldIndex> lgr_fields;
    std::vector<std::pair<int, std::string>> osh_fields;
};

void VtkOutput::set_fields(Omega_h::InputMap& pl) {
  auto dim = sim.dim();
  std::map<std::string, int> omega_h_adapt_tags = {
        {"quality", dim}, {"metric", 0}};
  std::map<std::string, std::pair<int, std::string>>
    omega_h_multi_dim_tags = {{"element class_id", {dim, "class_id"}},
      {"node class_dim", {0, "class_dim"}}, {"node local", {0, "local"}},
      {"node global", {0, "global"}}, {"element local", {dim, "local"}},
      {"element global", {dim, "global"}}};
  auto& field_names_in = pl.get_list("fields");
  for (int i = 0; i < field_names_in.size(); ++i) {
    auto field_name = field_names_in.get<std::string>(i);
    if (omega_h_adapt_tags.count(field_name)) {
      if (!sim.disc.mesh.has_tag(0, "metric"))
        Omega_h::add_implied_isos_tag(&sim.disc.mesh);
      osh_fields.push_back({omega_h_adapt_tags[field_name], field_name});
      continue;
    }
    if (omega_h_multi_dim_tags.count(field_name)) {
      osh_fields.push_back({omega_h_multi_dim_tags[field_name].first,
          omega_h_multi_dim_tags[field_name].second});
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
    lgr_fields.push_back(fi);
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

void VtkOutput::respond() {
  Omega_h::ScopedTimer timer("VtkOutput::respond");
}

void VtkOutput::out_of_line_virtual_method() {
}

Response* vtk_output_factory(
    Simulation& sim, std::string const&, Omega_h::InputMap& pl) {
  return new VtkOutput(sim, pl);
}

}  // namespace lgr
