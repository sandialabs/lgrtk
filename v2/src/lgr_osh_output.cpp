#include <lgr_osh_output.hpp>
#include <Omega_h_file.hpp>
#include <lgr_field_index.hpp>
#include <lgr_response.hpp>
#include <lgr_simulation.hpp>

namespace lgr {

struct OshOutput : public Response {
  std::vector<FieldIndex> field_indices;
  std::string prefix;
  OshOutput(Simulation& sim_in, Omega_h::InputMap& pl)
    :Response(sim_in, pl)
    ,prefix(pl.get<std::string>("prefix", "checkpoint_"))
  {
    for (auto& field_ptr : sim.fields.storage) {
      if ((field_ptr->remap_type != RemapType::NONE) &&
          (field_ptr->remap_type != RemapType::SHAPE)) {
        field_indices.push_back(sim.fields.find(field_ptr->long_name));
      }
    }
  }
  void out_of_line_virtual_method() override;
  void respond() override final {
    sim.disc.mesh.set_coords(sim.get(sim.position)); // linear specific!
    sim.fields.copy_to_omega_h(sim.disc, field_indices);
    auto path = prefix + std::to_string(sim.step) + ".osh";
    Omega_h::binary::write(path, &sim.disc.mesh);
    sim.fields.remove_from_omega_h(sim.disc, field_indices);
  }
};

void OshOutput::out_of_line_virtual_method() {}

Response* osh_output_factory(Simulation& sim, std::string const&, Omega_h::InputMap& pl) {
  return new OshOutput(sim, pl);
}

}
