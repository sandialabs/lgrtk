#include <Omega_h_file.hpp>
#include <Omega_h_profile.hpp>
#include <lgr_field_index.hpp>
#include <lgr_response.hpp>
#include <lgr_simulation.hpp>
#include <lgr_vtk_output.hpp>

namespace lgr {

struct VtkOutput : public Response {
  std::vector<FieldIndex> field_indices;
  Omega_h::vtk::Writer writer;
  Omega_h::TagSet tags;
  VtkOutput(Simulation& sim_in, Omega_h::InputMap& pl)
      : Response(sim_in, pl),
        writer(pl.get<std::string>("path", "lgr_viz"), &sim.disc.mesh,
            sim.dim(), sim.time, Omega_h::vtk::dont_compress) {
    auto stdim = std::size_t(sim.dim());
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
        if (!sim.disc.mesh.has_tag(0, "metric"))
          Omega_h::add_implied_isos_tag(&sim.disc.mesh);
        tags[omega_h_adapt_tags[field_name]].insert(field_name);
        continue;
      }
      if (omega_h_multi_dim_tags.count(field_name)) {
        tags[omega_h_multi_dim_tags[field_name].first].insert(
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
      if (support->subset->entity_type == NODES) {
        tags[0].insert(field_name);
      } else if (support->subset->entity_type == ELEMS) {
        if (support->on_points() && (sim.disc.points_per_ent(ELEMS) > 1)) {
          Omega_h_fail(
              "\"%s\" is on integration points, and there is more than one per "
              "element, "
              "and Dan hasn't coded support for visualizing that yet!\n",
              field_name.c_str());
        } else {
          tags[std::size_t(sim.dim())].insert(field_name);
        }
      } else {
        Omega_h_fail(
            "\"%s\" is not on nodes or elements, VTK can't visualize it!\n",
            field_name.c_str());
      }
      field_indices.push_back(fi);
    }
  }
  void out_of_line_virtual_method() override;
  void respond() override final {
    Omega_h::ScopedTimer timer("VtkOutput::respond");
    sim.disc.mesh.set_coords(sim.get(sim.position));  // linear specific!
    sim.fields.copy_to_omega_h(sim.disc, field_indices);
    writer.write(sim.step, sim.time, tags);
    sim.fields.remove_from_omega_h(sim.disc, field_indices);
  }
};

void VtkOutput::out_of_line_virtual_method() {}

Response* vtk_output_factory(
    Simulation& sim, std::string const&, Omega_h::InputMap& pl) {
  return new VtkOutput(sim, pl);
}

}  // namespace lgr
